namespace FoodOptimizer.Core;

/// <summary>Which menu command a key press is, from the Windows shortcut
/// lists in messages.json. A letter or digit matches its key wherever the
/// layout puts it; a symbol matches the character the layout types, so
/// Ctrl+= works on US keys, Ctrl++ on German ones and Ctrl+- on AZERTY.
/// Ctrl+Alt is AltGr on European keyboards and never routes: it types.</summary>
public sealed class KeyRouter
{
    const string Prefix = "Ctrl+";
    const int VkAdd = 0x6B, VkSubtract = 0x6D, VkNumpad0 = 0x60;

    readonly List<(string Id, string Key)> _bindings;

    KeyRouter(List<(string Id, string Key)> bindings) => _bindings = bindings;

    public static KeyRouter FromMessages(Messages m)
    {
        var bindings = new List<(string, string)>();
        foreach (var id in MenuIds.Commands)
        {
            foreach (var token in m.Keys(id))
            {
                var key = token.StartsWith(Prefix, StringComparison.Ordinal) ? token[Prefix.Length..] : "";
                if (key is not ("Add" or "Subtract") && key.Length != 1)
                    throw new ArgumentException("keys.windows: " + token, nameof(m));
                bindings.Add((id, key));
            }
        }
        return new KeyRouter(bindings);
    }

    /// <summary>The menu id for this key press, or null.</summary>
    public string? Route(int virtualKey, char? layoutChar, bool ctrl, bool shift, bool alt)
    {
        if (alt || !ctrl) return null;
        foreach (var (id, key) in _bindings)
            if (Matches(key, virtualKey, layoutChar)) return id;
        return null;
    }

    static bool Matches(string key, int vk, char? layoutChar)
    {
        if (key == "Add") return vk == VkAdd;
        if (key == "Subtract") return vk == VkSubtract;
        var c = key[0];
        if (char.IsAsciiLetter(c) || char.IsAsciiDigit(c))
        {
            var upper = char.ToUpperInvariant(c);
            if (vk == upper) return true;
            if (char.IsAsciiDigit(c) && vk == VkNumpad0 + (c - '0')) return true;
            return layoutChar is { } lc && char.ToUpperInvariant(lc) == upper;
        }
        return layoutChar == c;
    }
}
