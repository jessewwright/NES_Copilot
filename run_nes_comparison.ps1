# Run Python script in HDDM container

# Get the current directory path in Windows format
$currentDir = (Get-Item -Path ".\" -Verbose).FullName

# Convert Windows path to WSL/Linux format if needed
$wslPath = $currentDir -replace '\\', '/' -replace '^(.+):', '/mnt/$1'.ToLower()

# Run the comparison script in the HDDM container
docker run --rm `
  -v "${wslPath}:/workspace" `
  -w /workspace `
  --entrypoint python `
  hcp4715/hddm:latest `
  src/nes_copilot/compare_nes_hddm_empirical.py
