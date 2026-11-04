#requires -Version 7.2
# Gera o pacote assinado do eden-UWP: EdenUWP_<versao>_x64.appx, mais o VCLibs em Dependencies.
# A versao e a hora em UTC, entao cada pacote instala por cima do anterior.
# A chave de assinatura e criada na primeira execucao em .local (chave de teste, NAO vai para o GitHub).
[CmdletBinding()]
param([string]$Out = '')
$ErrorActionPreference = 'Stop'

$root  = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$local = Join-Path $root '.local'
$exe   = Join-Path $root 'build-uwp/bin/eden-uwp.exe'
$dist  = Join-Path $root 'dist/uwp'
if (!(Test-Path -LiteralPath $exe)) { throw "Falta o executavel: $exe (compile o alvo eden-uwp primeiro)" }

# Ferramentas do Windows SDK (versao mais nova instalada)
$sdk = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Directory |
    Where-Object { $_.Name -like '10.*' } |
    Sort-Object { [version]$_.Name } -Descending |
    ForEach-Object { Join-Path $_.FullName 'x64' } |
    Where-Object { Test-Path (Join-Path $_ 'makeappx.exe') } |
    Select-Object -First 1
if (!$sdk) { throw 'makeappx.exe nao encontrado no Windows SDK' }

$vclibs = "${env:ProgramFiles(x86)}\Microsoft SDKs\Windows Kits\10\ExtensionSDKs\Microsoft.VCLibs\14.0\Appx\Retail\x64\Microsoft.VCLibs.x64.14.00.appx"
if (!(Test-Path -LiteralPath $vclibs)) { throw "Falta o pacote VCLibs: $vclibs" }

# Chave de assinatura de teste (criada uma vez; o Publisher do manifesto e CN=eden-uwp)
New-Item -ItemType Directory -Force $local | Out-Null
$pfx = Join-Path $local 'eden-uwp-signing.pfx'
$cer = Join-Path $local 'eden-uwp-signing.cer'
$pfxPass = 'eden-uwp-dev'
if (!(Test-Path -LiteralPath $pfx)) {
    $cert = New-SelfSignedCertificate -Type Custom -Subject 'CN=eden-uwp' -KeyUsage DigitalSignature `
        -FriendlyName 'eden-UWP dev signing' -CertStoreLocation 'Cert:\CurrentUser\My' `
        -TextExtension @('2.5.29.37={text}1.3.6.1.5.5.7.3.3', '2.5.29.19={text}')
    Export-PfxCertificate -Cert $cert -FilePath $pfx -Password (ConvertTo-SecureString $pfxPass -AsPlainText -Force) | Out-Null
    Export-Certificate -Cert $cert -FilePath $cer | Out-Null
    Remove-Item -LiteralPath "Cert:\CurrentUser\My\$($cert.Thumbprint)"
}

if (!$Out) { $Out = Join-Path $local ('packages/' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')) }
$stage = Join-Path $Out 'layout'
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item -LiteralPath $exe -Destination $stage
Copy-Item -LiteralPath (Join-Path $dist 'Assets') -Destination $stage -Recurse
Copy-Item -LiteralPath (Join-Path $dist 'AppxManifest.xml') -Destination $stage

$now = [DateTime]::UtcNow
$version = '1.0.{0}.{1}' -f ($now.Year % 100 * 1000 + $now.DayOfYear), ([int]$now.ToString('HHmm'))
$manifest = Join-Path $stage 'AppxManifest.xml'
$content = (Get-Content -LiteralPath $manifest -Raw).Replace('Version="1.0.0.0"', "Version=`"$version`"")
[IO.File]::WriteAllText($manifest, $content, [Text.UTF8Encoding]::new($false))

$package = Join-Path $Out "EdenUWP_$version`_x64.appx"
$packOut = & "$sdk/makeappx.exe" pack /d $stage /p $package /o 2>&1
if ($LASTEXITCODE) { $packOut | Select-Object -Last 15; throw 'Falha ao criar o pacote' }
& "$sdk/signtool.exe" sign /q /fd SHA256 /f $pfx /p $pfxPass $package
if ($LASTEXITCODE) { throw 'Falha ao assinar o pacote' }
Remove-Item -LiteralPath $stage -Recurse -Force

New-Item -ItemType Directory -Force (Join-Path $Out 'Dependencies') | Out-Null
Copy-Item -LiteralPath $vclibs -Destination (Join-Path $Out 'Dependencies')

Write-Output "Pacote assinado criado: $version"
Write-Output "pacote=$package"
Write-Output "certificado=$cer"
