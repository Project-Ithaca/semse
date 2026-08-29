import Foundation

/// Instant calculator for the quick-actions layer. Uses a tiny recursive-descent
/// evaluator instead of NSExpression: NSExpression raises ObjC exceptions on
/// malformed input (uncatchable from pure Swift without an ObjC shim target),
/// and its format-string parser mis-handles `%` and `^`. This parser accepts
/// exactly the validated token set, so nothing can throw.
enum Calculator {
    private static let maxInputLength = 128

    static func action(for query: String) -> QuickAction? {
        guard query.count <= maxInputLength, looksLikeMath(query),
              let value = evaluate(query)
        else { return nil }
        let formatted = format(value)
        return QuickAction(
            id: "calc:\(query)",
            kind: .copyResult(formatted),
            title: "= \(formatted)",
            subtitle: "Press Return to copy",
            icon: nil,
            systemImage: "equal"
        )
    }

    /// Only digits, whitespace, and + - * / ( ) . % ^ — with at least one
    /// digit and one operator. Anything else is not treated as math.
    static func looksLikeMath(_ s: String) -> Bool {
        var hasDigit = false
        var hasOperator = false
        for ch in s {
            switch ch {
            case "0"..."9": hasDigit = true
            case "+", "-", "*", "/", "%", "^": hasOperator = true
            case "(", ")", ".": break
            default:
                if !ch.isWhitespace { return false }
            }
        }
        return hasDigit && hasOperator
    }

    static func evaluate(_ s: String) -> Double? {
        var parser = Parser(s)
        guard let v = parser.parse(), v.isFinite else { return nil }
        return v
    }

    static func format(_ v: Double) -> String {
        if v == v.rounded(), abs(v) < 1e15 {
            return String(Int64(v))
        }
        return String(format: "%.10g", v)
    }

    /// expr := term (('+'|'-') term)*
    /// term := unary (('*'|'/'|'%') unary)*
    /// unary := ('+'|'-')* power
    /// power := primary ('^' unary)?          (right-associative)
    /// primary := number | '(' expr ')'
    private struct Parser {
        private let chars: [Character]
        private var pos = 0

        init(_ s: String) {
            chars = Array(s.filter { !$0.isWhitespace })
        }

        mutating func parse() -> Double? {
            guard let v = expression(), pos == chars.count else { return nil }
            return v
        }

        private mutating func expression() -> Double? {
            guard var left = term() else { return nil }
            while let op = peek(), op == "+" || op == "-" {
                pos += 1
                guard let right = term() else { return nil }
                left = op == "+" ? left + right : left - right
            }
            return left
        }

        private mutating func term() -> Double? {
            guard var left = unary() else { return nil }
            while let op = peek(), op == "*" || op == "/" || op == "%" {
                pos += 1
                guard let right = unary() else { return nil }
                switch op {
                case "*": left *= right
                case "/": left /= right
                default: left = left.truncatingRemainder(dividingBy: right)
                }
            }
            return left
        }

        private mutating func unary() -> Double? {
            var negate = false
            while let c = peek(), c == "+" || c == "-" {
                if c == "-" { negate.toggle() }
                pos += 1
            }
            guard let v = power() else { return nil }
            return negate ? -v : v
        }

        private mutating func power() -> Double? {
            guard let base = primary() else { return nil }
            guard peek() == "^" else { return base }
            pos += 1
            guard let exponent = unary() else { return nil }
            return pow(base, exponent)
        }

        private mutating func primary() -> Double? {
            if peek() == "(" {
                pos += 1
                guard let v = expression(), peek() == ")" else { return nil }
                pos += 1
                return v
            }
            return number()
        }

        private mutating func number() -> Double? {
            let start = pos
            var sawDigit = false
            var sawDot = false
            while let c = peek() {
                if c.isNumber {
                    sawDigit = true
                } else if c == ".", !sawDot {
                    sawDot = true
                } else {
                    break
                }
                pos += 1
            }
            guard sawDigit else { return nil }
            return Double(String(chars[start..<pos]))
        }

        private func peek() -> Character? {
            pos < chars.count ? chars[pos] : nil
        }
    }
}
