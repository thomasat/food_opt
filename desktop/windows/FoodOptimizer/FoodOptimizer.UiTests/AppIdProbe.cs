using System.Runtime.InteropServices;

namespace FoodOptimizer.UiTests;

/// <summary>Reads the taskbar identity (PKEY_AppUserModel_ID) a window or a
/// shortcut carries.</summary>
static class AppIdProbe
{
    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPropertyStore
    {
        void GetCount(out uint count);
        void GetAt(uint index, out PropertyKey key);
        void GetValue(ref PropertyKey key, out PropVariant value);
        void SetValue(ref PropertyKey key, ref PropVariant value);
        void Commit();
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    struct PropertyKey
    {
        public Guid FormatId;
        public uint PropertyId;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct PropVariant
    {
        public ushort Type;
        public ushort Reserved1, Reserved2, Reserved3;
        public IntPtr Value;
        public IntPtr Value2;
    }

    [DllImport("shell32.dll")]
    static extern int SHGetPropertyStoreForWindow(IntPtr hwnd, ref Guid riid, out IPropertyStore store);

    [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
    static extern int SHGetPropertyStoreFromParsingName(string path, IntPtr bindContext, int flags, ref Guid riid, out IPropertyStore store);

    [DllImport("ole32.dll")]
    static extern int PropVariantClear(ref PropVariant value);

    static readonly Guid PropertyStoreId = new("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99");

    /// <summary>The window's own ID ("" when it sets none).</summary>
    public static string OfWindow(IntPtr hwnd)
    {
        var riid = PropertyStoreId;
        Assert.Equal(0, SHGetPropertyStoreForWindow(hwnd, ref riid, out var store));
        return Read(store);
    }

    /// <summary>The shortcut's ID ("" when it has none).</summary>
    public static string OfShortcut(string lnk)
    {
        var riid = PropertyStoreId;
        Assert.Equal(0, SHGetPropertyStoreFromParsingName(lnk, IntPtr.Zero, 0, ref riid, out var store));
        return Read(store);
    }

    static string Read(IPropertyStore store)
    {
        try
        {
            var key = new PropertyKey { FormatId = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), PropertyId = 5 };
            store.GetValue(ref key, out var value);
            try
            {
                return value.Type == 31 ? Marshal.PtrToStringUni(value.Value) ?? "" : "";     // VT_LPWSTR
            }
            finally { PropVariantClear(ref value); }
        }
        finally { Marshal.ReleaseComObject(store); }
    }
}
