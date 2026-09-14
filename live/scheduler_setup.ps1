# QuantLab 每日批处理 — Windows 任务计划注册脚本
#
# 用法（在 PowerShell 中执行）:
#   powershell -ExecutionPolicy Bypass -File live\scheduler_setup.ps1
#
# 注册后每天 15:30（收盘后）自动运行 live\run_daily.py：
#   增量更新数据 → 生成目标组合/调仓指令文件 → 月末调仓日执行（仅 simulate 模式）
# 默认不接真实柜台、不真实下单，绝对安全。
# 如需 QMT 真实下单：请人工在调仓日运行
#   python live\run_daily.py --qmt --confirm

$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot   # 项目根目录 D:\quantlab
$pythonExe = (Get-Command python).Source
if (-not $pythonExe) {
    throw "未找到 python，请先安装并加入 PATH"
}

$taskName = "QuantLab每日批处理"
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Output "已移除旧任务: $taskName"
}

$action = New-ScheduledTaskAction `
    -Execute $pythonExe `
    -Argument "live\run_daily.py" `
    -WorkingDirectory $projectDir

# 每天 15:30 触发；脚本内部会判断是否交易日/调仓日
$trigger = New-ScheduledTaskTrigger -Daily -At 15:30

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "QuantLab 每日批处理：增量更新数据、生成信号与调仓指令（不真实下单）" | Out-Null

Write-Output "已注册任务: $taskName"
Write-Output "  执行: $pythonExe live\run_daily.py"
Write-Output "  工作目录: $projectDir"
Write-Output "  时间: 每天 15:30（脚本自动跳过非交易日）"
Write-Output ""
Write-Output "验证: schtasks /Query /TN $taskName"
Write-Output "手动测试: & $pythonExe $projectDir\live\run_daily.py"
