default: build-debug

configure:
  cmake --preset default

build-release:
  cmake --build build --config RelWithDebInfo
alias br := build-release

build-debug:
  cmake --build build --config Debug

run IMAGE="./exr/cam7.exr": build-release
  # rm -f out/*.png
  ./build/RelWithDebInfo/filter "{{IMAGE}}"
alias r := run