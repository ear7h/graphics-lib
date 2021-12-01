with import <nixpkgs> {};

stdenv.mkDerivation {
  name = "game-engine";

  nativeBuildInputs = [
    gdb
  ];

  buildInputs = [
    xorg.libxcb
  ];

  # set the LD_LIBRARY_PATH so that the program can find wayland gui libs
  shellHook = ''
    export LD_LIBRARY_PATH=/run/opengl-driver/lib/:${
      lib.makeLibraryPath [ wayland libGL libxkbcommon xorg.libxcb ]
    }
  '';
}

