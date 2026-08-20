# 比赛启动器

运行：

```bash
cd /home/user/piper_ws
./start_competition_launcher.sh
```

也可以使用 `python3 competition_launcher.py`。请勿使用 `python2`：本程序需要 Python 3。

点击单项按钮会在同一个 `gnome-terminal` 窗口中新建一个标签页，效果等同于 `Ctrl+Shift+T`；该窗口会自动最小化。服务型命令在运行 5 秒后进程仍存活时，按钮变绿；CAN 配置脚本正常执行结束时变绿。红色表示启动进程提前退出或 CAN 脚本失败，具体日志在相应终端内。

CAN 配置已使用受限 sudo 规则授权默认 `start_can.sh` 调用，因此启动器中无需输入 sudo 密码。

“一键顺序启动”严格按 1 到 9 的顺序：当前一项成功后才启动下一项，任何一项失败都会停止后续启动。

若 ROS Master 已在运行，第 1 项会自动复用它，而不会重复启动 `roscore`。

一键启动按钮右侧可输入比赛速度，默认 `0.80 m/s`，它会作为第 9 项总控的 `target_speed`。比赛路线位于 `competition_launcher.py` 的第 9 项：`final02.csv`。后续添加其他启动命令时，只需在 `ITEMS` 列表中增加一项。
