// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Memory",
    platforms: [.macOS("26.0")],
    targets: [
        .executableTarget(
            name: "Memory",
            path: "Sources/Memory"
        ),
        .testTarget(
            name: "MemoryTests",
            dependencies: ["Memory"],
            path: "Tests/MemoryTests"
        )
    ]
)
