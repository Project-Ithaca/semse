import SwiftUI

/// The one row that stands between the user and everything older than two
/// weeks. Click (or Return while selected) to expand.
struct PastDisclosureRow: View {
    let count: Int
    let expanded: Bool
    let highlighted: Bool
    let onToggle: () -> Void

    var body: some View {
        Button(action: onToggle) {
            HStack(spacing: 10) {
                ZStack {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(Color.white.opacity(0.08))
                    Image(systemName: "clock.arrow.circlepath")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.white.opacity(0.7))
                }
                .frame(width: 22, height: 22)
                Text("Past")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.white.opacity(0.92))
                Text("\(count) older result\(count == 1 ? "" : "s")")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.tertiaryText)
                Spacer(minLength: 8)
                Image(systemName: "chevron.right")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundColor(Theme.tertiaryText)
                    .rotationEffect(.degrees(expanded ? 90 : 0))
            }
            .padding(.horizontal, 10)
            .frame(height: 36)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .fill(highlighted ? Color.white.opacity(0.06) : Color.clear)
            )
            .contentShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
        }
        .buttonStyle(.plain)
        .animation(.easeOut(duration: 0.15), value: expanded)
    }
}
