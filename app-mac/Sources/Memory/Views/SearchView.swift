import AppKit
import SwiftUI

struct SearchView: View {
    let onDismiss: () -> Void
    let onContentHeightChange: (CGFloat) -> Void

    @State private var query = ""
    @State private var response: SearchResponse?
    @State private var loading = false
    @State private var error: String?
    /// Single selection index across [top quick actions] → [semantic sources]
    /// → [bottom quick actions]. nil = nothing selected.
    @State private var selection: Int?
    /// Results split by age: anything older than Recency.recentWindowDays sits
    /// behind the collapsed "Past" row until the user opens it.
    @State private var recentSources: [SourceResult] = []
    @State private var pastSources: [SourceResult] = []
    @State private var showPast = false
    @StateObject private var quick = QuickActionsModel()

    @State private var debounceTask: Task<Void, Never>?
    @State private var currentTask: Task<Void, Never>?
    @State private var resultsContentHeight: CGFloat = 0
    @FocusState private var focused: Bool

    private let minQueryLength = 3
    private let debounceMs: UInt64 = 250
    // Max height for the results scroll area. The panel itself caps at ~60%
    // of the screen (see panelMaxHeight in AppDelegate); subtracting ~90pt
    // leaves room for the input row and the divider.
    private var scrollMaxHeight: CGFloat {
        let screenHeight = NSScreen.main?.visibleFrame.height ?? 900
        return max(300, screenHeight * 0.6 - 90)
    }

    var body: some View {
        VStack(spacing: 0) {
            searchInput
                .padding(.horizontal, 16)
                .padding(.vertical, 16)

            if showsBody {
                Divider().background(Theme.dividerLine)
                // Inner GeometryReader measures resultsContent's natural
                // height; the ScrollView frame is then min(measured, max) so
                // the panel grows with results up to scrollMaxHeight, then
                // scrolling kicks in inside the panel.
                ScrollView(.vertical, showsIndicators: true) {
                    resultsContent
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            GeometryReader { proxy in
                                Color.clear.preference(
                                    key: ResultsContentHeightKey.self,
                                    value: proxy.size.height
                                )
                            }
                        )
                }
                .frame(height: min(max(resultsContentHeight, 1), scrollMaxHeight))
                .onPreferenceChange(ResultsContentHeightKey.self) { newHeight in
                    resultsContentHeight = newHeight
                }
            }
        }
        .frame(width: 720)
        .fixedSize(horizontal: false, vertical: true)
        // Explicit drag strip along the top edge (sits inside the input row's
        // top padding) so the panel stays draggable even when SwiftUI content
        // fills the rest of the background.
        .overlay(alignment: .top) {
            PanelDragHandle()
                .frame(height: 12)
        }
        .background(
            GeometryReader { proxy in
                Color.clear.preference(key: ContentHeightKey.self, value: proxy.size.height)
            }
        )
        .onPreferenceChange(ContentHeightKey.self) { newHeight in
            // Hop to the next runloop tick so the panel resize doesn't run
            // inside the same layout pass that produced the new height.
            // Same-pass resize was breaking the chain on subsequent queries.
            DispatchQueue.main.async {
                onContentHeightChange(newHeight)
            }
        }
        .onAppear { focused = true }
        .onReceive(NotificationCenter.default.publisher(for: .focusSearchField)) { _ in
            focused = true
        }
        .onReceive(NotificationCenter.default.publisher(for: .clearSearchState)) { _ in
            // Fired by AppDelegate after the panel has stayed hidden long
            // enough — wipe the search so the next open is fresh.
            query = ""
            response = nil
            clearSources()
            error = nil
            selection = nil
            quick.update(query: "")
            currentTask?.cancel()
            debounceTask?.cancel()
        }
        .background(KeyMonitor(onEsc: handleEsc, onArrow: handleArrow, onReturn: handleReturn))
    }

    private var resultsContent: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !quick.topActions.isEmpty {
                quickGroup(quick.topActions, baseIndex: 0)
                if hasSemanticContent {
                    Divider().background(Theme.dividerLine)
                }
            }
            if let err = error {
                Text(err)
                    .font(.system(size: 12.5))
                    .foregroundColor(.red.opacity(0.85))
                    .padding(.horizontal, 4)
            }
            if loading && response == nil {
                ShimmerCards()
            }
            if let answer = response?.answer, !answer.isEmpty {
                SynthesisCard(answer: answer)
            }
            if let sources = response?.sources, !sources.isEmpty {
                sourceRows(recentSources, baseIndex: quick.topActions.count)
                if !pastSources.isEmpty {
                    PastDisclosureRow(
                        count: pastSources.count,
                        expanded: showPast,
                        highlighted: pastToggleIndex == selection,
                        onToggle: togglePast
                    )
                    .onHover { hovering in if hovering { selection = pastToggleIndex } }
                    if showPast {
                        sourceRows(pastSources, baseIndex: pastToggleIndex + 1)
                    }
                }
            } else if let r = response, r.sources.isEmpty, !loading {
                emptyResults
            }
            if !quick.bottomActions.isEmpty {
                if hasSemanticContent || !quick.topActions.isEmpty {
                    Divider().background(Theme.dividerLine)
                }
                quickGroup(quick.bottomActions, baseIndex: quick.topActions.count + sourceCount)
            }
            if let r = response {
                footer(r)
            }
        }
    }

    private func quickGroup(_ actions: [QuickAction], baseIndex: Int) -> some View {
        VStack(spacing: 2) {
            ForEach(Array(actions.enumerated()), id: \.element.id) { offset, action in
                let flatIdx = baseIndex + offset
                QuickActionRow(
                    action: action,
                    highlighted: flatIdx == selection,
                    copied: quick.copiedID == action.id,
                    onActivate: { activateQuick(action, revealInFinder: false) }
                )
                .onHover { hovering in if hovering { selection = flatIdx } }
            }
        }
    }

    @ViewBuilder
    private func sourceRows(_ sources: [SourceResult], baseIndex: Int) -> some View {
        ForEach(Array(sources.enumerated()), id: \.offset) { idx, src in
            let flatIdx = baseIndex + idx
            if src.source == "image" {
                ImageResultCard(result: src, highlighted: flatIdx == selection)
                    .onHover { hovering in if hovering { selection = flatIdx } }
            } else {
                SourceCard(result: src, highlighted: flatIdx == selection)
                    .onHover { hovering in if hovering { selection = flatIdx } }
            }
        }
    }

    /// Flat selection index of the "Past" row. Only meaningful when
    /// pastSources is non-empty.
    private var pastToggleIndex: Int {
        quick.topActions.count + recentSources.count
    }

    /// Rows the user can actually select right now — collapsed past results
    /// are not among them.
    private var sourceCount: Int {
        guard !pastSources.isEmpty else { return recentSources.count }
        return recentSources.count + 1 + (showPast ? pastSources.count : 0)
    }

    private func togglePast() {
        showPast.toggle()
        if !showPast, let sel = selection, sel > pastToggleIndex {
            // Collapsing removed the selected row — fall back to the toggle.
            selection = pastToggleIndex
        }
    }

    private var hasSemanticContent: Bool {
        loading || response != nil || error != nil
    }

    private var showsBody: Bool {
        hasSemanticContent || !quick.topActions.isEmpty || !quick.bottomActions.isEmpty
    }

    private var searchInput: some View {
        HStack(spacing: 12) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 16, weight: .light))
                .foregroundColor(Theme.tertiaryText.opacity(0.9))
            ZStack(alignment: .leading) {
                if query.isEmpty {
                    Text("Search your conversations…")
                        .font(.system(size: 17))
                        .foregroundColor(Theme.placeholder)
                }
                TextField("", text: $query)
                    .textFieldStyle(.plain)
                    .font(.system(size: 17))
                    .foregroundColor(.white)
                    .focused($focused)
                    .onChange(of: query) { _, newValue in
                        selection = nil
                        quick.update(query: newValue)
                        scheduleSearch(for: newValue)
                    }
            }
        }
    }

    private var emptyResults: some View {
        Text("No matches.")
            .font(.system(size: 12.5))
            .foregroundColor(Theme.tertiaryText)
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.vertical, 20)
    }

    private func footer(_ r: SearchResponse) -> some View {
        HStack {
            Spacer()
            Text("\(footerCount) · \(r.queryMs)ms")
                .font(.system(size: 10.5))
                .foregroundColor(Theme.tertiaryText.opacity(0.7))
        }
        .padding(.top, 4)
        .padding(.horizontal, 4)
    }

    // MARK: - Actions

    private func scheduleSearch(for value: String) {
        debounceTask?.cancel()
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count < minQueryLength {
            response = nil
            clearSources()
            error = nil
            loading = false
            currentTask?.cancel()
            return
        }
        debounceTask = Task {
            try? await Task.sleep(nanoseconds: debounceMs * 1_000_000)
            if Task.isCancelled { return }
            await runSearch(trimmed)
        }
    }

    private func runSearch(_ q: String) async {
        currentTask?.cancel()
        let task = Task {
            await MainActor.run { loading = true; error = nil }
            do {
                let resp = try await SearchClient.shared.search(query: q)
                if Task.isCancelled { return }
                await MainActor.run {
                    response = resp
                    let split = Recency.partition(resp.sources)
                    recentSources = split.recent
                    pastSources = split.past
                    showPast = false
                    loading = false
                }
            } catch {
                if Task.isCancelled { return }
                await MainActor.run {
                    self.error = error.localizedDescription
                    self.loading = false
                }
            }
        }
        currentTask = task
        await task.value
    }

    private func handleEsc() {
        if !query.isEmpty || response != nil {
            query = ""
            response = nil
            clearSources()
            error = nil
            selection = nil
        } else {
            onDismiss()
        }
    }

    private func handleArrow(_ dir: ArrowDirection) {
        let total = quick.topActions.count + sourceCount + quick.bottomActions.count
        guard total > 0 else { selection = nil; return }
        switch dir {
        case .down:
            selection = min(total - 1, (selection ?? -1) + 1)
        case .up:
            if let sel = selection {
                selection = sel <= 0 ? nil : sel - 1
            }
        }
    }

    /// Return activates the selected row; with nothing selected it activates
    /// the first quick action if present. Semantic source rows have no
    /// default activation, except the "Past" row, which expands. Cmd+Return
    /// on a file row reveals it in Finder.
    private func handleReturn(cmdHeld: Bool) {
        let top = quick.topActions
        let bottom = quick.bottomActions
        guard let sel = selection else {
            if let first = top.first { activateQuick(first, revealInFinder: cmdHeld) }
            return
        }
        if sel < top.count {
            activateQuick(top[sel], revealInFinder: cmdHeld)
        } else if !pastSources.isEmpty, sel == pastToggleIndex {
            togglePast()
        } else if sel >= top.count + sourceCount, sel - top.count - sourceCount < bottom.count {
            activateQuick(bottom[sel - top.count - sourceCount], revealInFinder: cmdHeld)
        }
    }

    /// With past results collapsed, a bare total ("8 sources") contradicts a
    /// panel showing one card — so name both buckets instead.
    private var footerCount: String {
        if !pastSources.isEmpty && !showPast {
            return "\(recentSources.count) shown · \(pastSources.count) past"
        }
        let total = recentSources.count + pastSources.count
        return "\(total) source\(total == 1 ? "" : "s")"
    }

    private func clearSources() {
        recentSources = []
        pastSources = []
        showPast = false
    }

    private func activateQuick(_ action: QuickAction, revealInFinder: Bool) {
        if quick.activate(action, revealInFinder: revealInFinder) {
            onDismiss()
        }
    }
}

enum ArrowDirection { case up, down }

extension Notification.Name {
    static let focusSearchField = Notification.Name("memory.focusSearchField")
    static let clearSearchState = Notification.Name("memory.clearSearchState")
}

/// SearchView reports its laid-out height through this preference so the
/// AppKit panel can resize. Replaces the old NSHostingView.intrinsicContentSize
/// KVO, which doesn't fire on SwiftUI re-layouts.
private struct ContentHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

/// Measures the natural height of the results-content stack so the
/// surrounding ScrollView can cap itself at scrollMaxHeight while still
/// shrinking to fit when results are short.
private struct ResultsContentHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

/// Invisible strip that initiates a window drag on mouse-down. Placed along
/// the panel's top edge so there is always a grabbable area, in addition to
/// isMovableByWindowBackground picking up empty background regions.
private struct PanelDragHandle: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView { DragView() }
    func updateNSView(_ nsView: NSView, context: Context) {}

    final class DragView: NSView {
        override func mouseDown(with event: NSEvent) {
            window?.performDrag(with: event)
        }
    }
}

/// Catches Esc and arrow keys at the window level so they work even when the
/// text field is focused.
private struct KeyMonitor: NSViewRepresentable {
    let onEsc: () -> Void
    let onArrow: (ArrowDirection) -> Void
    let onReturn: (Bool) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = MonitorView()
        view.onEsc = onEsc
        view.onArrow = onArrow
        view.onReturn = onReturn
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}

    final class MonitorView: NSView {
        var onEsc: (() -> Void)?
        var onArrow: ((ArrowDirection) -> Void)?
        var onReturn: ((Bool) -> Void)?
        private var monitor: Any?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            monitor.map { NSEvent.removeMonitor($0) }
            guard window != nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self = self else { return event }
                switch event.keyCode {
                case 53: self.onEsc?(); return nil          // esc
                case 125: self.onArrow?(.down); return nil  // arrow down
                case 126: self.onArrow?(.up); return nil    // arrow up
                case 36, 76:                                // return / keypad enter
                    self.onReturn?(event.modifierFlags.contains(.command))
                    return nil
                default: return event
                }
            }
        }

        deinit { monitor.map { NSEvent.removeMonitor($0) } }
    }
}

struct ShimmerCards: View {
    @State private var phase: CGFloat = -1

    var body: some View {
        VStack(spacing: 8) {
            ForEach(0..<3) { _ in
                shimmerRow
            }
        }
        .onAppear {
            withAnimation(.linear(duration: 1.4).repeatForever(autoreverses: false)) {
                phase = 1
            }
        }
    }

    private var shimmerRow: some View {
        HStack(spacing: 10) {
            Circle().fill(Color.white.opacity(0.06)).frame(width: 28, height: 28)
            VStack(alignment: .leading, spacing: 6) {
                RoundedRectangle(cornerRadius: 4).fill(Color.white.opacity(0.06)).frame(height: 10).frame(maxWidth: 140)
                RoundedRectangle(cornerRadius: 4).fill(Color.white.opacity(0.06)).frame(height: 10)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color.white.opacity(0.02)))
    }
}
