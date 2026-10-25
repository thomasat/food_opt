using System.Text.Json;
using System.Text.RegularExpressions;

namespace FoodOptimizer.Core;

/// <summary>One menu of the Windows menu bar: its message id and its items'
/// ids, in order; <see cref="Separator"/> marks a separator.</summary>
public sealed record MenuSpec(string Id, IReadOnlyList<string> Items)
{
    public const string Separator = "-";
}

/// <summary>desktop/shared/messages.json read for one OS: every string the
/// shell shows, by id, plus the menu layout and the shortcut keys. The shell
/// writes no screen text of its own (tests/test_windows_shell.py).</summary>
public sealed partial class Messages
{
    const string ResourceName = "messages.json";

    readonly Dictionary<string, string?> _texts;
    readonly Dictionary<string, IReadOnlyList<string>> _keys;

    Messages(Dictionary<string, string?> texts, Dictionary<string, IReadOnlyList<string>> keys,
             IReadOnlyList<MenuSpec> layout)
    {
        _texts = texts;
        _keys = keys;
        MenuLayout = layout;
        All = texts.Where(kv => kv.Value != null)
                   .ToDictionary(kv => kv.Key, kv => kv.Value!);
    }

    /// <summary>Every id with a text for this OS.</summary>
    public IReadOnlyDictionary<string, string> All { get; }

    /// <summary>menu_layout for this OS (empty when the file has none).</summary>
    public IReadOnlyList<MenuSpec> MenuLayout { get; }

    /// <summary>The bytes of the messages.json built into this assembly.</summary>
    public static byte[] EmbeddedBytes()
    {
        using var s = typeof(Messages).Assembly.GetManifestResourceStream(ResourceName)
            ?? throw new InvalidOperationException(ResourceName);
        using var m = new MemoryStream();
        s.CopyTo(m);
        return m.ToArray();
    }

    public static Messages LoadEmbedded()
    {
        using var s = new MemoryStream(EmbeddedBytes());
        return Load(s);
    }

    public static Messages Load(Stream utf8Json, string os = "windows")
    {
        using var doc = JsonDocument.Parse(utf8Json);
        var root = doc.RootElement;
        var texts = new Dictionary<string, string?>(StringComparer.Ordinal);
        var keys = new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal);
        foreach (var group in new[] { "messages", "menus" })
        {
            if (!root.TryGetProperty(group, out var list)) continue;
            foreach (var e in list.EnumerateArray())
            {
                var id = e.GetProperty("id").GetString()!;
                string? text = null;
                if (e.TryGetProperty("text", out var t) && t.TryGetProperty(os, out var v)
                    && v.ValueKind == JsonValueKind.String)
                    text = v.GetString();
                texts[id] = text;
                if (e.TryGetProperty("keys", out var k) && k.TryGetProperty(os, out var kl)
                    && kl.ValueKind == JsonValueKind.Array)
                    keys[id] = kl.EnumerateArray().Select(x => x.GetString()!).ToList();
            }
        }
        var layout = new List<MenuSpec>();
        if (root.TryGetProperty("menu_layout", out var ml) && ml.TryGetProperty(os, out var menus))
        {
            foreach (var m in menus.EnumerateArray())
                layout.Add(new MenuSpec(
                    m.GetProperty("menu").GetString()!,
                    m.GetProperty("items").EnumerateArray().Select(x => x.GetString()!).ToList()));
        }
        return new Messages(texts, keys, layout);
    }

    /// <summary>The text of <paramref name="id"/>; throws when the entry is
    /// absent or has no text for this OS.</summary>
    public string Text(string id) =>
        _texts.TryGetValue(id, out var t) && t != null
            ? t
            : throw new KeyNotFoundException("messages.json: " + id);

    /// <summary>The text with each {name} filled in; a placeholder left
    /// unfilled is an error, never shown.</summary>
    public string Format(string id, params (string Name, string Value)[] values)
    {
        var text = Text(id);
        foreach (var (name, value) in values)
            text = text.Replace("{" + name + "}", value, StringComparison.Ordinal);
        if (Placeholder().IsMatch(text))
            throw new FormatException("messages.json: " + id);
        return text;
    }

    /// <summary>The shortcut keys listed for a menu item on this OS.</summary>
    public IReadOnlyList<string> Keys(string id) =>
        _keys.TryGetValue(id, out var k) ? k : [];

    [GeneratedRegex(@"\{[^{}]*\}")]
    private static partial Regex Placeholder();
}
