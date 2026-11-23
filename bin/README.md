# Optional Local Binaries (`bin/`)

`HFWToolkit` uses Microsoft's DirectX Shader Compiler (`dxcompiler.dll`) to disassemble
Horizon Forbidden West's SM 6.6 DXIL shaders in the Material & Shader Deep-Dive Studio.

`HFWToolkit` automatically searches for `dxcompiler.dll` in this order:
1. `HFWToolkit/bin/dxcompiler.dll` (this folder)
2. `<HFW Game Dir>/tools/bin/x64/dxcompiler.dll`
3. Windows 10/11 SDK (`C:\Program Files (x86)\Windows Kits\10\bin\*\x64\dxcompiler.dll`)

If `hfw-toolkit doctor` reports `DXCompiler DLL: MISSING`, install the Windows 10/11 SDK
or download the latest release from [microsoft/DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler/releases)
and place `x64/dxcompiler.dll` inside this folder.
