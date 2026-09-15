# Melo.Core

MeloStudio 的跨平台 C++20 核心库。核心代码只表达领域模型、应用规则和纯计算；操作系统、UI、音频设备、进程与插件通信都位于外围适配器中。

## 设计边界

- `include/melo/core` 是公开 C++ API。
- `src` 是与操作系统无关的实现。
- 公共 API 不包含平台头文件。
- 时间单位和实体 ID 使用强类型，禁止裸整数混传。
- 时间区间统一使用半开区间 `[start, end)`。
- 对外 C ABI、IPC 宿主和平台适配器将在独立 target 中实现，不污染 `melo_core`。

## CLion 2026.2

直接在 CLion 中打开本目录。进入 `Settings | Build, Execution, Deployment | CMake`，启用导入的 `Melo.Core Debug` Preset，并选择 `MSVC-x64` 工具链。

可用目标：

- `melo_core`：静态核心库。
- `melo_core_tests`：无第三方依赖的基础测试。

## 命令行构建

```shell
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

安装后，其他 CMake 项目可以通过包配置引用：

```cmake
find_package(MeloCore CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE Melo::Core)
```

## 下一阶段

下一步按顺序加入 `domain`、`application`、`evaluation`，先完成项目模型、事务、revision、不可变快照和撤销/重做，再建立独立进程宿主。
