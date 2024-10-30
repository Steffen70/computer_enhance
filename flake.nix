{
  description = "A environment for working with Python, C and C#";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, ... }@inputs:
    inputs.flake-utils.lib.eachDefaultSystem (
      system:
      let
        unstable = import inputs.nixpkgs { inherit system; };

        # Create a custom Python environment with the necessary packages
        myPython = unstable.python312.withPackages (ps: with ps; [ numpy ]);
      in
      {
        devShell = unstable.mkShell {
          buildInputs = [
            unstable.nixfmt-rfc-style
            unstable.git
            myPython
            unstable.gcc
            unstable.glibc
            unstable.dotnet-sdk_8
          ];
        };
      }
    );
}
