import AppKit

// MARK: - Model

enum QuickActionKind {
    case openApp(URL)
    case copyResult(String)
    case openFile(URL)
    case openURL(URL)
}

struct QuickAction: Identifiable {
    let id: String
    let kind: QuickActionKind
    let title: String
    let subtitle: String?
    let icon: NSImage?
    let systemImage: String?
}

/// Computes instant, fully-local quick actions (app launcher, calculator,
/// file search, URL / web fallback) for the current query. Zero network.
@MainActor
final class QuickActionsModel: ObservableObject {
    @Published private(set) var topActions: [QuickAction] = []
    @Published private(set) var bottomActions: [QuickAction] = []
    @Published private(set) var copiedID: String?

    private let apps = AppIndex()
    private let files = FileSearcher()
    private var fileDebounce: Task<Void, Never>?
    private var currentQuery = ""
    private var appActions: [QuickAction] = []
    private var calcAction: QuickAction?
    private var fileActions: [QuickAction] = []

    private static let fileDebounceMs: UInt64 = 250
    private static let maxQuickTokens = 3
    private static let minMatchLength = 2

    init() {
        apps.onChanged = { [weak self] in self?.recomputeInstant() }
    }

    func update(query raw: String) {
        currentQuery = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        copiedID = nil
        fileDebounce?.cancel()
        files.cancel()
        fileActions = []
        recomputeInstant()
        scheduleFileSearch()
    }

    /// Panel closes for open-style actions; returns whether to dismiss.
    func activate(_ action: QuickAction, revealInFinder: Bool = false) -> Bool {
        switch action.kind {
        case .openApp(let url):
            NSWorkspace.shared.openApplication(at: url, configuration: NSWorkspace.OpenConfiguration())
            return true
        case .copyResult(let value):
            let pb = NSPasteboard.general
            pb.clearContents()
            pb.setString(value, forType: .string)
            copiedID = action.id
            let id = action.id
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: 1_200_000_000)
                if self?.copiedID == id { self?.copiedID = nil }
            }
            return false
        case .openFile(let url):
            if revealInFinder {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            } else {
                NSWorkspace.shared.open(url)
            }
            return true
        case .openURL(let url):
            NSWorkspace.shared.open(url)
            return true
        }
    }

    // MARK: - Assembly

    private var isShortQuery: Bool {
        let tokens = currentQuery.split(whereSeparator: { $0.isWhitespace })
        return !tokens.isEmpty && tokens.count <= Self.maxQuickTokens
    }

    private func recomputeInstant() {
        if currentQuery.isEmpty {
            calcAction = nil
            appActions = []
        } else {
            calcAction = Calculator.action(for: currentQuery)
            let wantsApps = isShortQuery && calcAction == nil
                && currentQuery.count >= Self.minMatchLength && !currentQuery.contains("/")
            appActions = wantsApps ? appMatches(currentQuery) : []
        }
        publish()
    }

    private func scheduleFileSearch() {
        guard isShortQuery, calcAction == nil, currentQuery.count >= Self.minMatchLength else { return }
        let q = currentQuery
        fileDebounce = Task { [weak self] in
            try? await Task.sleep(nanoseconds: Self.fileDebounceMs * 1_000_000)
            guard !Task.isCancelled, let self, self.currentQuery == q else { return }
            self.files.search(q) { [weak self] hits in
                guard let self, self.currentQuery == q else { return }
                self.fileActions = hits.map { hit in
                    QuickAction(
                        id: "file:\(hit.path)",
                        kind: .openFile(URL(fileURLWithPath: hit.path)),
                        title: hit.name,
                        subtitle: (hit.path as NSString).deletingLastPathComponent
                            .replacingOccurrences(of: NSHomeDirectory(), with: "~"),
                        icon: NSWorkspace.shared.icon(forFile: hit.path),
                        systemImage: nil
                    )
                }
                self.publish()
            }
        }
    }

    private func publish() {
        var top = appActions
        if let calc = calcAction { top.append(calc) }
        top.append(contentsOf: fileActions)
        topActions = top

        var bottom: [QuickAction] = []
        if let url = WebActions.urlAction(for: currentQuery) { bottom.append(url) }
        if let web = WebActions.webSearchAction(for: currentQuery) { bottom.append(web) }
        bottomActions = bottom
    }

    private func appMatches(_ query: String) -> [QuickAction] {
        apps.matches(for: query, limit: 2).map { entry in
            QuickAction(
                id: "app:\(entry.url.path)",
                kind: .openApp(entry.url),
                title: "Open \(entry.name)",
                subtitle: "Application",
                icon: NSWorkspace.shared.icon(forFile: entry.url.path),
                systemImage: nil
            )
        }
    }
}

// MARK: - App launcher index

@MainActor
final class AppIndex {
    struct Entry: Sendable {
        let name: String
        let url: URL
    }

    var onChanged: (() -> Void)?

    private var entries: [Entry] = []
    private var lastScan: Date = .distantPast
    private var scanInFlight = false
    private static let refreshInterval: TimeInterval = 300

    func matches(for query: String, limit: Int) -> [Entry] {
        scanIfStale()
        let q = query.lowercased()
        let scored: [(Entry, Int)] = entries.compactMap { entry in
            guard let s = Self.score(name: entry.name.lowercased(), query: q) else { return nil }
            return (entry, s)
        }
        return scored
            .sorted { $0.1 != $1.1 ? $0.1 > $1.1 : $0.0.name.count < $1.0.name.count }
            .prefix(limit)
            .map(\.0)
    }

    private static func score(name: String, query: String) -> Int? {
        if name.hasPrefix(query) { return 100 }
        if name.split(separator: " ").contains(where: { $0.hasPrefix(query) }) { return 80 }
        if name.contains(query) { return 60 }
        return nil
    }

    private func scanIfStale() {
        guard !scanInFlight, Date().timeIntervalSince(lastScan) > Self.refreshInterval else { return }
        scanInFlight = true
        let hadEntries = !entries.isEmpty
        Task.detached(priority: .userInitiated) { [weak self] in
            let found = AppIndex.scan()
            await self?.finishScan(with: found, notify: !hadEntries)
        }
    }

    private func finishScan(with found: [Entry], notify: Bool) {
        entries = found
        lastScan = Date()
        scanInFlight = false
        if notify { onChanged?() }
    }

    /// Scans the standard app folders one level deep (e.g. /Applications/Utilities).
    private nonisolated static func scan() -> [Entry] {
        let dirs = ["/Applications", "/System/Applications", NSHomeDirectory() + "/Applications"]
        let fm = FileManager.default
        var out: [Entry] = []
        var seen = Set<String>()

        func add(_ path: String) {
            let name = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
            guard !name.isEmpty, seen.insert(name.lowercased()).inserted else { return }
            out.append(Entry(name: name, url: URL(fileURLWithPath: path)))
        }

        for dir in dirs {
            guard let items = try? fm.contentsOfDirectory(atPath: dir) else { continue }
            for item in items {
                let path = dir + "/" + item
                if item.hasSuffix(".app") {
                    add(path)
                    continue
                }
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else { continue }
                for sub in (try? fm.contentsOfDirectory(atPath: path)) ?? [] where sub.hasSuffix(".app") {
                    add(path + "/" + sub)
                }
            }
        }
        return out
    }
}

// MARK: - File search (Spotlight index, local)

@MainActor
final class FileSearcher {
    struct Hit {
        let path: String
        let name: String
    }

    private var query: NSMetadataQuery?
    private var observer: NSObjectProtocol?
    private static let maxResults = 3

    func search(_ text: String, completion: @escaping @MainActor ([Hit]) -> Void) {
        cancel()
        let q = NSMetadataQuery()
        q.predicate = NSPredicate(
            format: "kMDItemFSName CONTAINS[cd] %@ OR kMDItemDisplayName CONTAINS[cd] %@",
            text, text
        )
        q.searchScopes = [NSMetadataQueryUserHomeScope]
        observer = NotificationCenter.default.addObserver(
            forName: .NSMetadataQueryDidFinishGathering, object: q, queue: .main
        ) { _ in
            MainActor.assumeIsolated { [weak self] in
                guard let self, let q = self.query else { return }
                q.disableUpdates()
                var hits: [Hit] = []
                for i in 0..<min(Self.maxResults, q.resultCount) {
                    guard let item = q.result(at: i) as? NSMetadataItem,
                          let path = item.value(forAttribute: NSMetadataItemPathKey) as? String
                    else { continue }
                    let name = (item.value(forAttribute: NSMetadataItemDisplayNameKey) as? String)
                        ?? (path as NSString).lastPathComponent
                    hits.append(Hit(path: path, name: name))
                }
                self.cancel()
                completion(hits)
            }
        }
        query = q
        q.start()
    }

    func cancel() {
        if let o = observer {
            NotificationCenter.default.removeObserver(o)
            observer = nil
        }
        query?.stop()
        query = nil
    }
}

// MARK: - URL / web fallback

enum WebActions {
    private static let knownTLDs: Set<String> = [
        "com", "net", "org", "io", "co", "dev", "app", "ai", "edu", "gov", "mil",
        "info", "xyz", "me", "tv", "sh", "gg", "fm", "to", "uk", "us", "ca", "de",
        "fr", "jp", "in", "au", "nz", "br", "ch", "nl", "se", "no", "es", "it",
    ]

    static func urlAction(for query: String) -> QuickAction? {
        let t = query.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty, !t.contains(" ") else { return nil }

        let lower = t.lowercased()
        if lower.hasPrefix("http://") || lower.hasPrefix("https://") {
            guard let u = URL(string: t), let host = u.host, host.contains(".") else { return nil }
            return make(url: u, display: host)
        }

        let core = t.split(separator: "/").first.map(String.init) ?? t
        let labels = core.lowercased().split(separator: ".", omittingEmptySubsequences: false)
        guard labels.count >= 2,
              let tld = labels.last, knownTLDs.contains(String(tld)),
              labels.allSatisfy({ !$0.isEmpty && $0.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" } }),
              let u = URL(string: "https://\(t)")
        else { return nil }
        return make(url: u, display: core)
    }

    static func webSearchAction(for query: String) -> QuickAction? {
        let t = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return nil }
        var comps = URLComponents(string: "https://duckduckgo.com/")!
        comps.queryItems = [URLQueryItem(name: "q", value: t)]
        guard let u = comps.url else { return nil }
        return QuickAction(
            id: "web:\(t)",
            kind: .openURL(u),
            title: "Search the web for \u{201C}\(t)\u{201D}",
            subtitle: "DuckDuckGo",
            icon: nil,
            systemImage: "globe"
        )
    }

    private static func make(url: URL, display: String) -> QuickAction {
        QuickAction(
            id: "url:\(url.absoluteString)",
            kind: .openURL(url),
            title: "Open \(display)",
            subtitle: "Website",
            icon: nil,
            systemImage: "safari"
        )
    }
}
