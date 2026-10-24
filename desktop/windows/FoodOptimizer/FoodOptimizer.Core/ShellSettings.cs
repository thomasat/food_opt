using System.Text;
using System.Text.Json;

namespace FoodOptimizer.Core;

/// <summary>settings.json in the state folder: zoom, window placement and
/// the version last opened. A bad field falls back to its default and the
/// rest is kept, so a hand-edited or truncated file never stops the app.</summary>
public sealed class ShellSettings
{
    public int Version { get; set; } = 1;
    public double Zoom { get; set; } = Core.Zoom.Default;
    public string? LastOpenedVersion { get; set; }
    public SavedWindow? Window { get; set; }

    public static ShellSettings Load(string path)
    {
        var s = new ShellSettings();
        try
        {
            var bytes = File.ReadAllBytes(path);
            var span = bytes.AsSpan();
            if (span.StartsWith(Encoding.UTF8.Preamble)) span = span[3..];
            using var doc = JsonDocument.Parse(span.ToArray());
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object) return s;
            if (Int(root, "version") is { } v) s.Version = v;
            if (root.TryGetProperty("zoom", out var z) && z.ValueKind == JsonValueKind.Number)
                s.Zoom = Core.Zoom.Clamp(z.GetDouble());
            if (root.TryGetProperty("lastOpenedVersion", out var lv) && lv.ValueKind == JsonValueKind.String)
                s.LastOpenedVersion = lv.GetString();
            if (root.TryGetProperty("window", out var w) && w.ValueKind == JsonValueKind.Object)
                s.Window = ReadWindow(w);
        }
        catch (Exception)
        {
            // Missing, unreadable or not JSON: whatever was read so far stands.
        }
        return s;
    }

    static int? Int(JsonElement o, string name) =>
        o.TryGetProperty(name, out var e) && e.ValueKind == JsonValueKind.Number && e.TryGetInt32(out var i)
            ? i : null;

    static SavedWindow? ReadWindow(JsonElement w)
    {
        var monitor = w.TryGetProperty("monitor", out var m) && m.ValueKind == JsonValueKind.String
            ? m.GetString() : null;
        if (string.IsNullOrEmpty(monitor)) return null;
        if (Int(w, "left") is not { } left || Int(w, "top") is not { } top
            || Int(w, "width") is not { } width || Int(w, "height") is not { } height)
            return null;
        if (width <= 0 || height <= 0) return null;
        var maximized = w.TryGetProperty("maximized", out var mx) && mx.ValueKind == JsonValueKind.True;
        return new SavedWindow(monitor, left, top, width, height, maximized);
    }

    byte[] ToJson()
    {
        using var buffer = new MemoryStream();
        using (var j = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            j.WriteStartObject();
            j.WriteNumber("version", Version);
            j.WriteNumber("zoom", Zoom);
            if (LastOpenedVersion != null) j.WriteString("lastOpenedVersion", LastOpenedVersion);
            if (Window is { } w)
            {
                j.WriteStartObject("window");
                j.WriteString("monitor", w.Monitor);
                j.WriteNumber("left", w.Left);
                j.WriteNumber("top", w.Top);
                j.WriteNumber("width", w.Width);
                j.WriteNumber("height", w.Height);
                j.WriteBoolean("maximized", w.Maximized);
                j.WriteEndObject();
            }
            j.WriteEndObject();
        }
        buffer.WriteByte((byte)'\n');
        return buffer.ToArray();
    }

    /// <summary>Writes the file atomically (temp file, then replace);
    /// false when it could not. Never throws.</summary>
    public bool Save(string path)
    {
        var tmp = path + ".tmp";
        try
        {
            File.WriteAllBytes(tmp, ToJson());
        }
        catch (Exception)
        {
            return false;
        }
        for (var attempt = 0; attempt < 3; attempt++)
        {
            try
            {
                File.Move(tmp, path, overwrite: true);
                return true;
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                Thread.Sleep(50);
            }
            catch (Exception)
            {
                break;
            }
        }
        try { File.Delete(tmp); } catch (Exception) { }
        return false;
    }
}
