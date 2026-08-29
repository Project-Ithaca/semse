import XCTest
@testable import Memory

/// SourceResult only has a decoding initializer, so fixtures go through JSON —
/// which doubles as a check that the wire shape still decodes.
private func makeSource(
    source: String = "imessage",
    dateStart: String,
    dateEnd: String,
    snippet: String = ""
) -> SourceResult {
    let json = """
    {"source": "\(source)", "contact_names": ["Jerry Yan"],
     "date_start": "\(dateStart)", "date_end": "\(dateEnd)",
     "score": 1.0, "messages": [], "snippet": "\(snippet)"}
    """
    return try! JSONDecoder().decode(SourceResult.self, from: Data(json.utf8))
}

private let utc: Calendar = {
    var cal = Calendar(identifier: .gregorian)
    cal.timeZone = TimeZone(secondsFromGMT: 0)!
    return cal
}()

private let now = utc.date(from: DateComponents(year: 2026, month: 8, day: 29, hour: 12))!

private func iso(daysAgo: Double) -> String {
    let date = now.addingTimeInterval(-daysAgo * 86_400)
    let f = DateFormatter()
    f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
    f.locale = Locale(identifier: "en_US_POSIX")
    f.timeZone = TimeZone(secondsFromGMT: 0)
    return f.string(from: date)
}

final class RecencyParseTests: XCTestCase {
    func testParsesNaiveUTCString() {
        let d = Recency.parse("2026-08-15T09:30:00")
        XCTAssertNotNil(d)
        XCTAssertEqual(utc.component(.day, from: d!), 15)
        XCTAssertEqual(utc.component(.hour, from: d!), 9)
    }

    func testParsesFractionalSecondsAndOffsets() {
        XCTAssertNotNil(Recency.parse("2026-08-15T09:30:00.123Z"))
        XCTAssertNotNil(Recency.parse("2026-08-15T09:30:00Z"))
        XCTAssertNotNil(Recency.parse("2026-08-15T09:30:00+02:00"))
    }

    func testParsesDateOnly() {
        XCTAssertNotNil(Recency.parse("2026-08-15"))
    }

    func testRejectsEmptyAndGarbage() {
        XCTAssertNil(Recency.parse(""))
        XCTAssertNil(Recency.parse("not a date"))
    }
}

final class RecencyWindowTests: XCTestCase {
    func testTodayIsRecent() {
        XCTAssertTrue(Recency.isRecent(iso(daysAgo: 0), now: now))
    }

    func testJustInsideWindowIsRecent() {
        XCTAssertTrue(Recency.isRecent(iso(daysAgo: 13), now: now))
    }

    func testBoundaryDayIsRecent() {
        // The cutoff is the start of the day 14 days back, so a stamp from
        // midday that day still counts as recent.
        XCTAssertTrue(Recency.isRecent(iso(daysAgo: 14), now: now))
    }

    func testOlderThanWindowIsPast() {
        XCTAssertFalse(Recency.isRecent(iso(daysAgo: 15), now: now))
        XCTAssertFalse(Recency.isRecent(iso(daysAgo: 400), now: now))
    }

    func testFutureDatesAreRecent() {
        XCTAssertTrue(Recency.isRecent(iso(daysAgo: -5), now: now))
    }

    func testUnparseableDatesStayVisible() {
        XCTAssertTrue(Recency.isRecent("", now: now))
        XCTAssertTrue(Recency.isRecent("garbage", now: now))
    }
}

final class RecencyPartitionTests: XCTestCase {
    func testSplitsByAge() {
        let sources = [
            makeSource(dateStart: iso(daysAgo: 1), dateEnd: iso(daysAgo: 1), snippet: "a"),
            makeSource(dateStart: iso(daysAgo: 30), dateEnd: iso(daysAgo: 30), snippet: "b"),
            makeSource(dateStart: iso(daysAgo: 5), dateEnd: iso(daysAgo: 5), snippet: "c"),
        ]
        let split = Recency.partition(sources, now: now, alwaysVisible: 0)
        XCTAssertEqual(split.recent.map(\.snippet), ["a", "c"])
        XCTAssertEqual(split.past.map(\.snippet), ["b"])
    }

    func testPreservesRelevanceOrderWithinBuckets() {
        let sources = [
            makeSource(dateStart: iso(daysAgo: 100), dateEnd: iso(daysAgo: 100), snippet: "old1"),
            makeSource(dateStart: iso(daysAgo: 2), dateEnd: iso(daysAgo: 2), snippet: "new1"),
            makeSource(dateStart: iso(daysAgo: 500), dateEnd: iso(daysAgo: 500), snippet: "old2"),
            makeSource(dateStart: iso(daysAgo: 3), dateEnd: iso(daysAgo: 3), snippet: "new2"),
        ]
        let split = Recency.partition(sources, now: now, alwaysVisible: 0)
        XCTAssertEqual(split.recent.map(\.snippet), ["new1", "new2"])
        XCTAssertEqual(split.past.map(\.snippet), ["old1", "old2"])
    }

    func testSpanningChunkCountsByItsNewestMoment() {
        // A calendar group running from 40 days ago to 3 days ago is recent.
        let spanning = makeSource(
            source: "calendar", dateStart: iso(daysAgo: 40), dateEnd: iso(daysAgo: 3)
        )
        let split = Recency.partition([spanning], now: now, alwaysVisible: 0)
        XCTAssertEqual(split.recent.count, 1)
        XCTAssertTrue(split.past.isEmpty)
    }

    func testFallsBackToStartWhenEndMissing() {
        let noEnd = makeSource(dateStart: iso(daysAgo: 200), dateEnd: "")
        XCTAssertEqual(Recency.partition([noEnd], now: now, alwaysVisible: 0).past.count, 1)
    }

    func testUndatedResultStaysVisible() {
        let undated = makeSource(dateStart: "", dateEnd: "")
        XCTAssertEqual(Recency.partition([undated], now: now, alwaysVisible: 0).recent.count, 1)
    }

    func testEmptyInput() {
        let split = Recency.partition([], now: now, alwaysVisible: 0)
        XCTAssertTrue(split.recent.isEmpty)
        XCTAssertTrue(split.past.isEmpty)
    }

    func testAllPastLeavesRecentEmpty() {
        let sources = (0..<3).map {
            makeSource(dateStart: iso(daysAgo: Double(30 + $0)), dateEnd: iso(daysAgo: Double(30 + $0)))
        }
        let split = Recency.partition(sources, now: now, alwaysVisible: 0)
        XCTAssertTrue(split.recent.isEmpty)
        XCTAssertEqual(split.past.count, 3)
    }
}

/// The top results stay on screen regardless of age; only the tail collapses.
final class RecencyPinningTests: XCTestCase {
    private func allOld(_ count: Int) -> [SourceResult] {
        (0..<count).map {
            makeSource(dateStart: iso(daysAgo: Double(100 + $0)),
                       dateEnd: iso(daysAgo: Double(100 + $0)),
                       snippet: "s\($0)")
        }
    }

    func testTopThreeSurviveEvenWhenAllAreOld() {
        let split = Recency.partition(allOld(8), now: now)
        XCTAssertEqual(split.recent.map(\.snippet), ["s0", "s1", "s2"])
        XCTAssertEqual(split.past.map(\.snippet), ["s3", "s4", "s5", "s6", "s7"])
    }

    func testFewerResultsThanPinCountLeavesNothingPast() {
        let split = Recency.partition(allOld(2), now: now)
        XCTAssertEqual(split.recent.count, 2)
        XCTAssertTrue(split.past.isEmpty)
    }

    func testRecentResultsPastThePinStillShow() {
        var sources = allOld(3)
        sources.append(makeSource(dateStart: iso(daysAgo: 1), dateEnd: iso(daysAgo: 1),
                                  snippet: "fresh"))
        sources.append(makeSource(dateStart: iso(daysAgo: 300), dateEnd: iso(daysAgo: 300),
                                  snippet: "ancient"))
        let split = Recency.partition(sources, now: now)
        XCTAssertEqual(split.recent.map(\.snippet), ["s0", "s1", "s2", "fresh"])
        XCTAssertEqual(split.past.map(\.snippet), ["ancient"])
    }

    func testDefaultPinCountIsThree() {
        XCTAssertEqual(Recency.alwaysVisibleCount, 3)
    }
}
