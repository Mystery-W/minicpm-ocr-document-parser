param(
    [string]$Python = 'py',
    [ValidateSet('huggingface','modelscope')][string]$ModelSource = 'huggingface'
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root
$venv=Join-Path $root '.venv'
$venvPython=Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    if (Test-Path -LiteralPath $venv) { throw '.venv 已存在但不完整；未修改，请手动检查。' }
    if ($Python -eq 'py') { & py -3.11 -m venv $venv } else { & $Python -m venv $venv }
    if ($LASTEXITCODE -ne 0) { throw '创建项目独立环境失败。' }
}
& $venvPython -c "import sys; assert sys.version_info[:2] == (3,11), '本版本依赖锁定基于 Python 3.11'"
if ($LASTEXITCODE -ne 0) { throw '需要 Python 3.11。没有修改其他环境。' }
& $venvPython -m pip install -r (Join-Path $root 'requirements-windows.txt')
if ($LASTEXITCODE -ne 0) { throw '依赖安装失败。' }
& $venvPython (Join-Path $root 'setup_assets.py') --model-source $ModelSource
if ($LASTEXITCODE -ne 0) { throw '模型或推理程序准备失败；重新运行可继续下载。' }
Write-Host '安装结束。运行 启动.bat，再打开 http://127.0.0.1:7865/'
