{pkgs}: {
  deps = [
    pkgs.python310Packages.flask
    pkgs.python311Packages.py
    pkgs.xsimd
    pkgs.libxcrypt
    pkgs.glibcLocales
    pkgs.tk
    pkgs.tcl
    pkgs.qhull
    pkgs.pkg-config
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.ghostscript
    pkgs.freetype
    pkgs.ffmpeg-full
    pkgs.cairo
  ];
}
