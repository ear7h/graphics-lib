with import <nixpkgs> {};

stdenv.mkDerivation {
  name = "game-engine";

  nativeBuildInputs = [
    gdb
  ];

  # set the LD_LIBRARY_PATH so that the program can find wayland gui libs
  shellHook = ''
    export LD_LIBRARY_PATH=${
      lib.makeLibraryPath [ wayland libGL libxkbcommon ]
    }
  '';
}

