import Foundation

private func log(_ message: String) {
    FileHandle.standardError.write(Data("[Semse backend] \(message)\n".utf8))
}

/// Boots the local stack (ollama + FastAPI) if it isn't already running,
/// mirroring scripts/start_semse.command. Everything happens off the main
/// thread; the panel opens immediately and SearchClient surfaces errors
/// while the backend warms up.
enum BackendLauncher {
    private static let defaultRoot = "/Users/tarunyadgirkar/TarunsCode/semse"
    private static let apiHealthURL = URL(string: "http://localhost:8000/health")!
    private static let ollamaVersionURL = URL(string: "http://localhost:11434/api/version")!

    static var repoRoot: URL {
        let path = ProcessInfo.processInfo.environment["SEMSE_ROOT"] ?? defaultRoot
        return URL(fileURLWithPath: path)
    }

    static func ensureStackRunning() {
        Task.detached(priority: .utility) {
            async let llm: Void = ensureOllama()
            async let api: Void = ensureAPI()
            _ = await (llm, api)
        }
    }

    // MARK: - Ollama

    private static func ensureOllama() async {
        if await isReachable(ollamaVersionURL) {
            log("ollama already running")
            return
        }
        guard let binary = findOllama() else {
            log("ollama binary not found — LLM synthesis will be unavailable")
            return
        }
        log("starting ollama serve (\(binary.path))")
        launchDetached(executable: binary, arguments: ["serve"], workingDirectory: nil, logFile: nil)
        await waitUntilReachable(ollamaVersionURL, attempts: 30, name: "ollama")
    }

    private static func findOllama() -> URL? {
        let candidates = ["/opt/homebrew/bin/ollama", "/usr/local/bin/ollama"]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        let which = Process()
        which.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        which.arguments = ["which", "ollama"]
        let pipe = Pipe()
        which.standardOutput = pipe
        which.standardError = FileHandle.nullDevice
        do {
            try which.run()
            which.waitUntilExit()
        } catch {
            return nil
        }
        guard which.terminationStatus == 0 else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let path = String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path)
    }

    // MARK: - API

    private static func ensureAPI() async {
        if await isReachable(apiHealthURL) {
            log("api already running")
            return
        }
        let root = repoRoot
        let uvicorn = root.appendingPathComponent(".venv/bin/uvicorn")
        guard FileManager.default.isExecutableFile(atPath: uvicorn.path) else {
            log("uvicorn not found at \(uvicorn.path) — is SEMSE_ROOT correct?")
            return
        }
        let logDir = root.appendingPathComponent("indexer/data")
        try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
        log("starting api (uvicorn api.main:app --port 8000)")
        launchDetached(
            executable: uvicorn,
            arguments: ["api.main:app", "--port", "8000"],
            workingDirectory: root,
            logFile: logDir.appendingPathComponent("api.log")
        )
        await waitUntilReachable(apiHealthURL, attempts: 60, name: "api")
    }

    // MARK: - Helpers

    private static func launchDetached(
        executable: URL,
        arguments: [String],
        workingDirectory: URL?,
        logFile: URL?
    ) {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        if let workingDirectory { process.currentDirectoryURL = workingDirectory }
        process.standardInput = FileHandle.nullDevice
        if let logFile {
            if !FileManager.default.fileExists(atPath: logFile.path) {
                FileManager.default.createFile(atPath: logFile.path, contents: nil)
            }
            if let handle = try? FileHandle(forWritingTo: logFile) {
                handle.seekToEndOfFile()
                process.standardOutput = handle
                process.standardError = handle
            } else {
                process.standardOutput = FileHandle.nullDevice
                process.standardError = FileHandle.nullDevice
            }
        } else {
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
        }
        do {
            try process.run()
        } catch {
            log("failed to launch \(executable.lastPathComponent): \(error.localizedDescription)")
        }
    }

    /// Any HTTP response counts as "up" — /health reports degraded states
    /// with non-200 codes, but the server itself is running.
    private static func isReachable(_ url: URL) async -> Bool {
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return response is HTTPURLResponse
        } catch {
            return false
        }
    }

    private static func waitUntilReachable(_ url: URL, attempts: Int, name: String) async {
        for _ in 0..<attempts {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            if await isReachable(url) {
                log("\(name) is up")
                return
            }
        }
        log("\(name) did not come up after \(attempts)s — searches will show a connection error until it does")
    }
}
