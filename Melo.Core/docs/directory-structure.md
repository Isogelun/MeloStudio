# Melo.Core 目录结构与职责

> 状态：工程规范。本文描述 `Melo.Core/` 当前目录以及后续扩展时应采用的目录边界。
> 核心原则是：一份跨平台 C++ 源码、多平台分别编译；操作系统差异只能进入外围适配器。

## 1. 总体结构

当前已经建立的目录如下：

```text
Melo.Core/
├── cmake/                       CMake 包配置模板
├── docs/                        架构与工程规范文档
├── include/melo/core/           对外公开的 C++ API
├── src/                         跨平台核心实现
├── tests/                       自动化测试
├── .clang-format                C++ 格式规范
├── .editorconfig                通用文本格式规范
├── CMakeLists.txt               项目构建入口
├── CMakePresets.json            共享构建预设
└── README.md                    项目入口说明
```

随着功能增长，目标结构扩展为：

```text
Melo.Core/
├── adapters/                    操作系统和外部设施适配器
│   ├── audio/
│   ├── filesystem/
│   ├── ipc/
│   └── plugins/
├── benchmarks/                  性能测试与固定测量场景
├── cmake/                       CMake 模块和包配置
├── conformance/                 面向公开契约的黑盒一致性测试
├── docs/                        架构、协议和工程文档
├── hosts/                       加载核心库的可执行程序
│   ├── cli/
│   └── desktop/
├── include/melo/core/           稳定的公共 C++ 头文件
├── spec/                        与实现语言无关的规范源
│   ├── project-format/
│   ├── protocol/
│   └── semantics/
├── src/                         跨平台核心私有实现
│   ├── application/
│   ├── assets/
│   ├── audio/
│   ├── domain/
│   ├── evaluation/
│   ├── presentation/
│   ├── scheduler/
│   ├── storage/
│   └── synthesis/
└── tests/
    ├── integration/
    └── unit/
```

没有实际内容时不要提前创建空目录；新增模块时按本文定义落位。

## 2. 源码目录

### `include/melo/core/`

存放其他 C++ target 可以包含的公共 API。这些头文件随库一起安装，并通过
`Melo::Core` CMake target 暴露。

当前文件：

| 文件 | 作用 |
| --- | --- |
| `core.hpp` | 聚合头文件，为简单使用方提供统一入口 |
| `id.hpp` | `ProjectId`、`TrackId`、`PartId`、`NoteId` 等强类型 ID |
| `time.hpp` | `ProjectTick`、`SampleFrame`、`Seconds` 和半开时间区间 |
| `version.hpp` | 核心库的语义版本信息 |

放入这里的条件：

1. 类型或函数确实需要被库使用方访问。
2. 接口在 Windows、Linux、macOS 上具有相同语义。
3. 不包含 Win32、POSIX、音频设备或 UI 框架头文件。
4. 不暴露核心内部容器布局、线程对象和缓存实现。

仅供实现内部使用的头文件不能为了“方便 include”放进公共目录，应放在对应的
`src/<module>/` 下。

### `src/`

存放 `melo_core` 静态库的私有实现。这里的代码必须是标准 C++，不能依赖具体操作系统。

当前只有 `version.cpp`，用于提供可链接的版本查询。业务实现开始后按职责拆分：

| 子目录 | 负责内容 | 不负责内容 |
| --- | --- | --- |
| `domain/` | 工程实体、不变量、强类型时间、纯编辑规则 | JSON、线程、磁盘、设备、UI |
| `application/` | `ProjectSession`、命令、事务、revision、撤销重做 | 具体文件系统和 IPC |
| `evaluation/` | Tempo 换算、曲线插值、峰值查询、有效轨判定 | 任务排队和项目修改 |
| `presentation/` | 视口筛选、几何投影、渲染数据组装 | 主题、控件树和实际绘制 |
| `scheduler/` | 失效合并、依赖、优先级、取消和工作预算 | 领域数据的直接修改 |
| `synthesis/` | 合成输入准备、任务版本、产物验收 | 直接调用 Python 或修改工程 |
| `assets/` | PCM、特征、峰值等大块不可变产物的抽象 | 工程编辑语义 |
| `audio/` | 播放计划、跨平台混音算法、实时规则 | WASAPI、ALSA、CoreAudio |
| `storage/` | 工程保存/加载的用例和抽象接口 | 平台路径对话框和原生文件句柄 |

模块之间保持单向依赖：

```text
domain
   ↑
application
   ↑
evaluation / presentation / scheduler / synthesis
   ↑
adapters / hosts
```

下层模块不能反向包含上层模块。特别是 `domain` 不得依赖其他业务模块。

## 3. 外围目录

### `adapters/`（后续创建）

存放对核心抽象接口的具体实现，是平台差异和第三方 SDK 唯一允许集中的地方。

| 子目录 | 典型内容 |
| --- | --- |
| `audio/` | WASAPI、CoreAudio、ALSA/PipeWire 或跨平台音频库封装 |
| `filesystem/` | 原子保存、文件锁、路径和权限处理 |
| `ipc/` | Named Pipe、Unix Domain Socket、共享内存和协议传输 |
| `plugins/` | 插件进程启动、监督、动态库或工作进程通信 |

如果必须使用 `#ifdef _WIN32`、`__linux__` 或 `__APPLE__`，原则上只能出现在这里或
`hosts/` 的组合入口中，不能扩散到 `include/melo/core` 和核心业务实现。

### `hosts/`（后续创建）

存放可执行程序入口。宿主只负责组装对象、选择适配器和管理生命周期，不承载领域规则。

| 子目录 | 作用 |
| --- | --- |
| `desktop/` | `melostudio-core` 独立核心进程，供桌面 UI 通过 IPC 调用 |
| `cli/` | headless 命令入口，用于自动化、诊断和一致性测试 |

移动端或不允许子进程的平台可以直接链接 `Melo::Core`，不必使用这些宿主。

## 4. 规范与验证目录

### `tests/`

存放跟随源码一起构建的自动化测试。当前 `core_tests.cpp` 是零第三方依赖的基础测试，
验证安装前最基本的公共类型和链接行为。

功能增加后拆成：

| 子目录 | 作用 |
| --- | --- |
| `unit/` | 单个领域规则、时间换算、插值和数据结构测试 |
| `integration/` | 多模块组合、保存往返、任务取消和播放计划测试 |

测试代码可以访问测试夹具，但不能为了测试方便而扩大生产公共 API。

### `conformance/`（后续创建）

存放从进程或公开 API 外部观察核心行为的黑盒测试，验证协议、项目格式和语义契约。
它和 `tests/` 的区别是：`tests/` 验证 C++ 实现内部，`conformance/` 验证任何合格核心实现。

### `benchmarks/`（后续创建）

存放可重复的性能场景、固定输入数据和统计口径。基准结果不能依赖开发者个人工程，
也不能把平均值当作实时性能的唯一结论。音频场景至少记录分位数、最大耗时和 underrun。

### `spec/`（后续创建）

这是独立于 C++ 类定义的规范源。

| 子目录 | 作用 |
| --- | --- |
| `protocol/` | JSON Schema、消息帧、版本协商和二进制布局 |
| `project-format/` | 工程文件格式、schema version 和迁移规则 |
| `semantics/` | 时间、区间、插值、撤销和调度的规范行为 |

协议和工程格式不能从 C++ struct 内存布局直接推导。C++ DTO 可以从规范生成，但 C++ 类型本身
不是跨语言契约的唯一真相。

## 5. 构建与文档目录

### `cmake/`

存放 CMake 辅助模块和安装包模板。当前 `MeloCoreConfig.cmake.in` 用于生成
`MeloCoreConfig.cmake`，让安装后的使用方能够这样引用：

```cmake
find_package(MeloCore CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE Melo::Core)
```

业务源码、平台实现和测试夹具都不应放进这个目录。

### `docs/`

存放需要长期维护的设计依据和工程规则：

- `README.md`：文档索引。
- `core-architecture.md`：完整核心架构、线程、数据所有权和阶段计划。
- `directory-structure.md`：本文，规定代码与资源放置位置。
- `v3-architecture-decision.md`：分层和进程边界的决策记录。
- `v3-render-protocol-design.md`：渲染协议设计。
- `v3-架构决策图.png`：原始架构决策图。

代码行为改变后，如果影响公开契约、模块职责或线程模型，必须同步修改对应文档。

## 6. 根文件

| 文件 | 作用 | 是否提交 |
| --- | --- | --- |
| `CMakeLists.txt` | 定义项目、`melo_core`、测试、安装和导出规则 | 是 |
| `CMakePresets.json` | 团队共享的 Debug/Release 配置 | 是 |
| `CMakeUserPresets.json` | 开发者本机编译器路径和私有配置 | 否 |
| `.clang-format` | C++ 自动格式化规则，CLion 可直接读取 | 是 |
| `.editorconfig` | 编码、换行、缩进和尾随空格规则 | 是 |
| `README.md` | 构建入口、目标说明和快速开始 | 是 |

## 7. 生成目录

以下目录由 CLion、CMake 或安装验证生成，不属于源码：

| 目录 | 来源 | 处理方式 |
| --- | --- | --- |
| `.idea/` | CLion 项目状态和个人工作区配置 | 忽略，不作为架构目录 |
| `cmake-build-*/` | CLion 默认 CMake 构建目录 | 忽略，可随时重新生成 |
| `out/build/` | CMake Presets 构建输出 | 忽略，可随时重新生成 |
| `out/install/` | 本地安装和包引用验证输出 | 忽略，可随时重新生成 |

任何源码、规范、测试夹具或手写配置都不能放在生成目录中，否则清理构建缓存时会丢失。

## 8. 新代码放置判据

添加文件前按以下顺序判断：

1. 是公开且跨平台的 C++ 接口吗？放 `include/melo/core/`。
2. 是纯领域或纯计算实现吗？放 `src/` 对应模块。
3. 是否调用操作系统或第三方外部设施？放 `adapters/`。
4. 是否只是启动和组装核心？放 `hosts/`。
5. 是实现内部测试还是公开契约测试？分别放 `tests/` 或 `conformance/`。
6. 是机器生成的输出吗？放构建目录且不得提交。

如果一个文件同时符合多项，通常表示职责混合，应先拆分而不是任选一个目录。
