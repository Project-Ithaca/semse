import Foundation

/// Splits results into "recent" and "past" so the panel only shows the last
/// two weeks by default. Everything older sits behind one collapsed row the
/// user clicks to reveal.
enum Recency {
    static let recentWindowDays = 14

    private static let utcCalendar: Calendar = {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(secondsFromGMT: 0)!
        return cal
    }()

    private static let naiveFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(secondsFromGMT: 0)
        return f
    }()

    private static let dateOnlyFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(secondsFromGMT: 0)
        return f
    }()

    /// The API sends naive-UTC ISO strings (no offset suffix); anything with
    /// an explicit offset still parses through the ISO8601 path.
    static func parse(_ iso: String) -> Date? {
        if iso.isEmpty { return nil }
        if let d = naiveFormatter.date(from: String(iso.prefix(19))) { return d }
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = f.date(from: iso) { return d }
        f.formatOptions = [.withInternetDateTime]
        if let d = f.date(from: iso) { return d }
        return dateOnlyFormatter.date(from: String(iso.prefix(10)))
    }

    /// A date is recent when it falls within the last `recentWindowDays`
    /// calendar days (UTC). Future dates count as recent; an unparseable or
    /// missing date also counts as recent, so nothing is silently hidden.
    static func isRecent(_ iso: String, now: Date = Date()) -> Bool {
        guard let date = parse(iso) else { return true }
        guard let cutoff = utcCalendar.date(
            byAdding: .day, value: -recentWindowDays, to: utcCalendar.startOfDay(for: now)
        ) else { return true }
        return date >= cutoff
    }

    /// Partitions results by recency, preserving relevance order within each
    /// bucket. A result is dated by `dateEnd` — its newest moment — so a chunk
    /// that spans months counts as recent when it ran into the last two weeks.
    static func partition(
        _ sources: [SourceResult], now: Date = Date()
    ) -> (recent: [SourceResult], past: [SourceResult]) {
        var recent: [SourceResult] = []
        var past: [SourceResult] = []
        for source in sources {
            let stamp = source.dateEnd.isEmpty ? source.dateStart : source.dateEnd
            if isRecent(stamp, now: now) {
                recent.append(source)
            } else {
                past.append(source)
            }
        }
        return (recent, past)
    }
}
