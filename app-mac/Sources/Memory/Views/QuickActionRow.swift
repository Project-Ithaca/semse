import AppKit
import SwiftUI

/// Compact (~36pt) Spotlight-style row for a quick action.
struct QuickActionRow: View {
    let action: QuickAction
    let highlighted: Bool
    let copied: Bool
    let onActivate: () -> Void

    var body: some View {
        Button(action: onActivate) {
            HStack(spacing: 10) {
                iconView
                VStack(alignment: .leading, spacing: 0) {
                    Text(action.title)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.white.opacity(0.92))
                        .lineLimit(1)
                    if let subtitle = action.subtitle, !subtitle.isEmpty {
                        Text(subtitle)
                            .font(.system(size: 10.5))
                            .foregroundColor(Theme.tertiaryText)
                            .lineLimit(1)
                    }
                }
                Spacer(minLength: 8)
                trailing
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
    }

    @ViewBuilder
    private var iconView: some View {
        if let icon = action.icon {
            Image(nsImage: icon)
                .resizable()
                .frame(width: 22, height: 22)
        } else {
            ZStack {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Color.white.opacity(0.08))
                Image(systemName: action.systemImage ?? "arrow.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.white.opacity(0.7))
            }
            .frame(width: 22, height: 22)
        }
    }

    @ViewBuilder
    private var trailing: some View {
        if copied {
            Text("Copied")
                .font(.system(size: 10.5, weight: .semibold))
                .foregroundColor(Theme.imessageGreen)
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(Capsule().fill(Theme.imessageGreen.opacity(0.15)))
        } else if highlighted {
            Text("\u{21A9}")
                .font(.system(size: 11))
                .foregroundColor(Theme.tertiaryText)
        }
    }
}
