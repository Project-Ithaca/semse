import SwiftUI

struct SourceCard: View {
    let result: SourceResult
    let highlighted: Bool

    @State private var expanded = false

    private let collapsedShown = 3
    private let collapsedLines = 6

    /// How this card's body is laid out, derived from the wire `source` tag.
    /// Unknown tags fall through to `.generic` — plain text lines, never a
    /// broken chat-bubble layout.
    private enum RenderStyle {
        case chat, mail, calendar, reminders, generic
    }

    private var renderStyle: RenderStyle {
        switch result.source {
        case "imessage": return .chat
        case "mail": return .mail
        case "calendar": return .calendar
        case "reminders": return .reminders
        default: return .generic
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            bodyContent
            if hasMore && !expanded {
                expandButton
            } else if expanded && hasMore {
                collapseButton
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(highlighted ? Color.white.opacity(0.06) : Color.white.opacity(0.02))
        )
    }

    private var header: some View {
        HStack(spacing: 10) {
            switch renderStyle {
            case .chat, .mail:
                Avatar(name: primaryName, contactKey: primaryKey, size: 28)
            case .calendar:
                glyphBadge(systemName: "calendar", tint: Theme.calendarRed)
            case .reminders:
                glyphBadge(systemName: "checklist", tint: Theme.remindersOrange)
            case .generic:
                glyphBadge(systemName: "doc.text.magnifyingglass", tint: Color.white.opacity(0.55))
            }
            VStack(alignment: .leading, spacing: 1) {
                Text(displayTitle)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.white.opacity(0.92))
                    .lineLimit(1)
                Text(metaLine)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.tertiaryText)
                    .lineLimit(1)
            }
            Spacer()
        }
    }

    private func glyphBadge(systemName: String, tint: Color) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(tint.opacity(0.18))
            Image(systemName: systemName)
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(tint)
        }
        .frame(width: 28, height: 28)
    }

    @ViewBuilder
    private var bodyContent: some View {
        switch renderStyle {
        case .chat, .mail:
            messagesView
        case .calendar:
            if !chunkLines.isEmpty {
                linesBody(accent: Theme.calendarRed, bullet: nil)
            }
        case .reminders:
            if !chunkLines.isEmpty {
                linesBody(accent: nil, bullet: "circle")
            }
        case .generic:
            if !chunkLines.isEmpty {
                linesBody(accent: nil, bullet: nil)
            }
        }
    }

    private var messagesView: some View {
        let (visible, hiddenAbove, hiddenBelow) = visibleWindow()
        return VStack(spacing: 4) {
            if hiddenAbove > 0 && !expanded {
                contextRow(text: "\(hiddenAbove) earlier message\(hiddenAbove == 1 ? "" : "s")")
            }
            ForEach(Array(visible.enumerated()), id: \.offset) { idx, msg in
                if renderStyle == .mail {
                    MailRow(message: msg, expanded: expanded)
                } else {
                    MessageBubble(
                        message: msg,
                        showSender: idx == 0 || visible[idx - 1].sender != msg.sender
                    )
                }
            }
            if hiddenBelow > 0 && !expanded {
                contextRow(text: "\(hiddenBelow) later message\(hiddenBelow == 1 ? "" : "s")")
            }
        }
    }

    /// Body for calendar / reminders / unknown sources: the chunk's text
    /// rendered as plain lines, optionally with an event accent bar
    /// (calendar) or a checklist bullet (reminders).
    private func linesBody(accent: Color?, bullet: String?) -> some View {
        let lines = expanded ? chunkLines : Array(chunkLines.prefix(collapsedLines))
        return HStack(alignment: .top, spacing: 8) {
            if let accent {
                RoundedRectangle(cornerRadius: 1.5)
                    .fill(accent.opacity(0.8))
                    .frame(width: 3)
            }
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        if let bullet {
                            Image(systemName: bullet)
                                .font(.system(size: 10))
                                .foregroundColor(Theme.remindersOrange)
                        }
                        Text(line)
                            .font(.system(size: 12.5))
                            .foregroundColor(Theme.secondaryText)
                            .fixedSize(horizontal: false, vertical: true)
                            .textSelection(.enabled)
                    }
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.white.opacity(0.03))
        )
    }

    /// Center the visible window on the best-matching message (or the chunk
    /// tail if none is flagged). Returns (visibleSlice, hiddenBefore, hiddenAfter).
    private func visibleWindow() -> ([ChunkMessage], Int, Int) {
        if expanded { return (result.messages, 0, 0) }
        let count = result.messages.count
        if count <= collapsedShown { return (result.messages, 0, 0) }
        let anchor = result.messages.firstIndex(where: { $0.isBestMatch }) ?? (count - 1)
        // Window of `collapsedShown` messages centered on the anchor, clamped.
        let half = collapsedShown / 2
        var start = max(0, anchor - half)
        let end = min(count, start + collapsedShown)
        if end - start < collapsedShown {
            start = max(0, end - collapsedShown)
        }
        return (Array(result.messages[start..<end]), start, count - end)
    }

    private var expandButton: some View {
        Button(action: { withAnimation(.easeInOut(duration: 0.18)) { expanded = true } }) {
            HStack(spacing: 6) {
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 10, weight: .semibold))
                Text(expandLabel)
                    .font(.system(size: 11.5, weight: .medium))
            }
            .foregroundColor(Theme.accent)
            .padding(.vertical, 5)
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.white.opacity(0.04))
            )
        }
        .buttonStyle(.plain)
    }

    private var collapseButton: some View {
        Button(action: { withAnimation(.easeInOut(duration: 0.18)) { expanded = false } }) {
            HStack(spacing: 6) {
                Image(systemName: "chevron.up")
                    .font(.system(size: 10, weight: .semibold))
                Text("Collapse")
                    .font(.system(size: 11.5, weight: .medium))
            }
            .foregroundColor(Theme.tertiaryText)
            .padding(.vertical, 5)
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.plain)
    }

    private func contextRow(text: String) -> some View {
        HStack {
            line
            Text(text)
                .font(.system(size: 10.5, weight: .medium))
                .foregroundColor(Theme.tertiaryText)
            line
        }
        .padding(.vertical, 2)
    }

    private var line: some View {
        Rectangle().fill(Theme.dividerLine).frame(height: 1)
    }

    // MARK: - Derived

    private var hasMore: Bool {
        switch renderStyle {
        case .chat, .mail: return result.messages.count > collapsedShown
        case .calendar, .reminders, .generic: return chunkLines.count > collapsedLines
        }
    }

    private var expandLabel: String {
        switch renderStyle {
        case .chat, .mail:
            return "Show full conversation (\(result.messages.count) messages)"
        case .calendar, .reminders, .generic:
            return "Show all \(chunkLines.count) lines"
        }
    }

    /// The chunk's raw text split into display lines — used by the calendar,
    /// reminders, and generic bodies. Falls back to `snippet` when the server
    /// sent no per-message breakdown.
    private var chunkLines: [String] {
        let text = result.messages.isEmpty
            ? result.snippet
            : result.messages.map(\.text).joined(separator: "\n")
        return text
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    private var displayTitle: String {
        // Calendar/reminders chunks carry the event/list name in `subject`;
        // prefer it as the header for those (and any unknown) sources.
        switch renderStyle {
        case .calendar, .reminders, .generic:
            if let subject = result.subject, !subject.isEmpty { return subject }
            if let chat = result.chatTitle, !chat.isEmpty { return chat }
        case .chat, .mail:
            if let chat = result.chatTitle, !chat.isEmpty { return chat }
            if let subject = result.subject, !subject.isEmpty { return subject }
        }
        let names = result.contactNames.prefix(2).joined(separator: ", ")
        return names.isEmpty ? SourceLabels.label(for: result.source) : names
    }

    private var primaryName: String {
        result.contactNames.first ?? "Unknown"
    }

    private var primaryKey: String? {
        result.messages.first(where: { !$0.isFromMe })?.contactKey
    }

    private var metaLine: String {
        let label = SourceLabels.label(for: result.source)
        let date = formatDate(result.dateStart)
        return date.isEmpty ? label : "\(label) · \(date)"
    }
}

enum SourceLabels {
    static func label(for source: String) -> String {
        switch source {
        case "imessage": return "iMessage"
        case "mail": return "Mail"
        case "calendar": return "Calendar"
        case "reminders": return "Reminders"
        case "image": return "Image"
        case "hyperspell": return "Hyperspell"
        default: return source.capitalized
        }
    }
}

func formatDate(_ iso: String) -> String {
    if iso.isEmpty { return "" }
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    var date = f.date(from: iso)
    if date == nil {
        f.formatOptions = [.withInternetDateTime]
        date = f.date(from: iso)
    }
    if date == nil {
        let alt = DateFormatter()
        alt.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        alt.locale = Locale(identifier: "en_US_POSIX")
        date = alt.date(from: iso)
    }
    guard let d = date else { return String(iso.prefix(10)) }
    let out = DateFormatter()
    out.dateStyle = .medium
    return out.string(from: d)
}
