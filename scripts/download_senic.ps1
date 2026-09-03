param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
$RawRoot = Join-Path $ProjectRoot "data\raw\SeNic"
$SubjectRoot = Join-Path $RawRoot "subjects"
$ArchiveDir = Join-Path $RawRoot "archives"
$TempRepo = Join-Path $RawRoot "_sparse_h0_h5_repo"
$LogDir = Join-Path $ProjectRoot "logs"
$Log = Join-Path $LogDir "00_download_senic.txt"
New-Item -ItemType Directory -Force -Path $SubjectRoot,$ArchiveDir,$LogDir | Out-Null

function Test-RarFile([string]$Path) {
    if (-not (Test-Path $Path)) { return $false }
    if ((Get-Item $Path).Length -lt 1MB) { return $false }
    $Stream = [System.IO.File]::OpenRead($Path)
    try {
        $Header = New-Object byte[] 4
        [void]$Stream.Read($Header, 0, 4)
    }
    finally { $Stream.Dispose() }
    return ([System.Text.Encoding]::ASCII.GetString($Header)).StartsWith("Rar!")
}

function Extract-Rar([string]$Archive,[string]$Destination) {
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Write-Host "Extracting $(Split-Path $Archive -Leaf) -> $Destination" -ForegroundColor Yellow
    & tar.exe -xf $Archive -C $Destination
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe could not extract $Archive. Install a recent Windows bsdtar/7-Zip if needed."
    }
}

& {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "SeNic h0-h29 download and extraction" -ForegroundColor Cyan
    Write-Host "Project: $ProjectRoot" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
        throw "Git was not found. Install Git for Windows before running this script."
    }

    Write-Host "[1/2] h0-h5 sparse checkout" -ForegroundColor Cyan
    if (-not (Test-Path (Join-Path $TempRepo ".git"))) {
        if (Test-Path $TempRepo) { Remove-Item $TempRepo -Recurse -Force }
        git clone --filter=blob:none --no-checkout https://github.com/BoZhuBo/SeNic.git $TempRepo
        if ($LASTEXITCODE -ne 0) { throw "Git clone failed." }
    }
    Push-Location $TempRepo
    try {
        git sparse-checkout init --cone
        git sparse-checkout set h0 h1 h2 h3 h4 h5
        git checkout main
        if ($LASTEXITCODE -ne 0) { throw "Sparse checkout failed." }
    }
    finally { Pop-Location }

    foreach ($N in 0..5) {
        $Subject = "h$N"
        $SourceDir = Join-Path $TempRepo $Subject
        $Destination = Join-Path $SubjectRoot $Subject
        New-Item -ItemType Directory -Force -Path $Destination | Out-Null
        Get-ChildItem $SourceDir -File -Filter "Angle_*.xlsx" -ErrorAction SilentlyContinue |
            ForEach-Object { Copy-Item $_.FullName (Join-Path $Destination $_.Name) -Force }
        Get-ChildItem $SourceDir -File -Filter "*.txt" -ErrorAction SilentlyContinue |
            ForEach-Object { Copy-Item $_.FullName (Join-Path $Destination $_.Name) -Force }
        $Archives = @(Get-ChildItem $SourceDir -Recurse -File -Filter "*.rar" | Sort-Object FullName)
        if ($Archives.Count -eq 0) { Write-Warning "$Subject: no RAR files found in sparse checkout." }
        foreach ($Archive in $Archives) { Extract-Rar $Archive.FullName $Destination }
    }

    Write-Host "[2/2] h6-h29 direct archives" -ForegroundColor Cyan
    foreach ($N in 6..29) {
        $Subject = "h$N"
        $Archive = Join-Path $ArchiveDir "$Subject.rar"
        $Temp = "$Archive.part"
        if (-not (Test-RarFile $Archive)) {
            Remove-Item $Temp -Force -ErrorAction SilentlyContinue
            $Urls = @(
                "https://github.com/BoZhuBo/SeNic/raw/refs/heads/main/$Subject.rar",
                "https://raw.githubusercontent.com/BoZhuBo/SeNic/main/$Subject.rar",
                "https://gitee.com/bozhubo/SeNic/raw/main/$Subject.rar"
            )
            $Ok = $false
            foreach ($Url in $Urls) {
                Write-Host "Downloading $Subject from $Url" -ForegroundColor Yellow
                & curl.exe -L --fail --silent --show-error --retry 8 --retry-all-errors `
                    --retry-delay 3 --connect-timeout 30 --max-time 1800 `
                    -A "Mozilla/5.0" -o $Temp $Url
                if ($LASTEXITCODE -eq 0 -and (Test-RarFile $Temp)) {
                    Move-Item $Temp $Archive -Force
                    $Ok = $true
                    break
                }
                Remove-Item $Temp -Force -ErrorAction SilentlyContinue
            }
            if (-not $Ok) {
                Write-Warning "$Subject download failed; continue and rerun later."
                continue
            }
        }
        Extract-Rar $Archive (Join-Path $SubjectRoot $Subject)
    }

    Write-Host "Download/extraction stage finished." -ForegroundColor Green
    Write-Host "Run: .\.venv\Scripts\python.exe .\scripts\audit_dataset.py" -ForegroundColor Green
} 2>&1 | Tee-Object -FilePath $Log
