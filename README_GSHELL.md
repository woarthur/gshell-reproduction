# GShell 论文复现代码

对应用户提供的论文 **GShell: Decoupled 3DGS for Optically Complex Scenes with Finite-thickness Transparent Shells**，Pacific Graphics 2026 稿件编号 1235。

本实现依据论文第 3、4 节及第 5.1 节超参数编写，没有作者实现可供逐项比对。交付范围是方法代码、训练/渲染/导出入口和实现说明；没有执行训练、渲染、数值测试或实验论证，也没有验证论文报告的效果和速度。

## 基础仓库

基础为 [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)，提交 `54c035f7834b564019656c3e3fcc3646292f727d`。工作目录原本已有官方源码，本次已拉取远端信息，确认该提交与远端 `main` 一致，递归子模块齐全。

复用上游的球谐函数、SH 系数布局、COLMAP 格式读取以及高斯参数与增密思路。GShell 使用新的有限线段积分器，不调用原始投影光栅器；原有 `train.py`、`render.py`、CUDA 子模块保持原实现。新增代码遵循根目录 `LICENSE.md`，上游文件保留原版权声明。

## 论文与实现对应

| 论文内容 | 代码位置 | 实现 |
| --- | --- | --- |
| 双独立 SDF，4.1、4.6 | `gshell/encoding.py`, `gshell/fields.py` | 16 层、每层 2 特征、16 到 2048 分辨率、每层最多 2^19 项的 HashGrid；两层 64 单元 MLP |
| 方向代理，式 13 | `gshell/fields.py` | 四阶 SH 编码，三层 128 单元，sigmoid RGB；不接收位置或视图编号 |
| 折射/反射/吸收，式 1-5 | `gshell/optics.py` | Snell、Schlick、全反射判定、RGB Beer-Lambert |
| 可微 SDF 求交，4.6 | `gshell/tracing.py` | 无梯度球面追踪和符号穿越细化；最终交点用隐函数定理连接梯度 |
| 高斯表示，4.1 | `gshell/gaussians.py` | 位置、对数尺度、四元数、SH 颜色；softplus 单位长度密度 |
| 解析积分，式 6-9、11 | `gshell/integrator.py` | 有限线段与 Mahalanobis 支撑相交后计算 erf 积分；统一排序、前到后合成 |
| 完整光学模型，式 12 | `gshell/renderer.py` | 前外、前内、后内、后外四次求交；首表面反射和双壁吸收 |
| HDR 环境，4.4、4.6 | `gshell/environment.py` | 对数辐亮度贴图，周期双线性采样，128×256 → 256×512 → 512×1024，TV 正则 |
| 三阶段，式 14-16 | `gshell/model.py`, `gshell/training.py`, `gshell/losses.py` | 明确设置可学习参数、阶段交接和损失；20k/10k/40k 步 |
| 增密与裁剪，4.5 | `gshell/gaussians.py` | 小高斯克隆、大高斯分裂、密度/尺度裁剪、Adam 状态迁移 |
| 可编辑组件，5.7 | `gshell/export.py`, `render_gshell.py` | 两个独立壳体网格、内部高斯、HDR；组件隔离与环境替换 |

完整合成关系为：

```text
C = R1 E(dr1) + T1 Af T2 [Cint + Tres T3 Ab T4 E(d4)]
alpha_i = 1 - exp(-softplus(raw_density_i) * J_i)
```

内部光线从前内界面到后内界面的有限段上积分，不使用屏幕空间 opacity。不同高斯分块之间也进行统一深度排序，不各自独立合成。

## 阶段行为

1. **Stage I**：优化外 SDF、方向代理和全局材质。命中当前外壳的射线按折射方向查询代理；未命中射线按原方向查询。损失包含 RGB、外 SDF Eikonal 和法线平滑。
2. **Stage II**：冻结外 SDF 和材质，移除代理，仅优化环境。使用未命中射线，以及反射占比较高的外表面射线；不追踪透射内部内容。
3. **Stage III**：冻结外 SDF、环境和材质；将外 SDF 权重独立复制给内 SDF，并只在初始化时向内偏移。随后内表面独立学习，不保持平行偏移关系。优化内 SDF 与内部高斯，使用完整光学模型、RGB、SSIM、Eikonal、内部约束和尺度正则。

冻结场的参数不会更新，但 Stage III 对射线起点、方向以及交点法线的梯度仍会穿过外场和环境查询，供内表面优化使用。所有阶段均不需要前景掩码。

## 论文未充分说明之处与本实现选择

- **Stage I 吸收与颜色合成**：式 13 只定义方向颜色，正文却称该阶段估计吸收，而真实玻璃路径长度此时未知。本实现用同一个方向代理提供反射/透射颜色，以 Fresnel 混合，并采用 `bootstrap_absorption_length=0.05` 的名义路径让吸收参数参与优化。它不约束最终壳厚，且无法保证吸收可独立辨识。设为 0 可去掉该名义衰减。
- **Stage II 反射监督**：普通 RGB 像素含反射与透射混合，论文未提供分量分离算法。本实现选取 `R1 >= reflection_threshold`（默认 0.5）且非掠射的命中射线，将其 RGB 近似作为反射项监督，未选中的透射像素不参与环境拟合。该近似会受残余透射污染，不等于拿到了纯反射真值；将阈值设为 1 可仅用未命中射线。
- **初始化及归一化**：外场采用半径 0.8 的解析球体加可学习残差，以建立初始求交；内场只在第三阶段开始时复制和偏移。默认输入场景已以原点为中心，壳位于半径 1.5 的追踪域内。实际数据需设置 `scene_center` 和 `scene_scale`，不从透明场景的稀疏点自动推断壳体尺寸。
- **内部点初始化**：COLMAP 点仅在当前内 SDF 内采纳；不足部分通过腔内均匀拒绝采样补充，颜色初始化为 0.5。论文没有给出初始化点数、学习率或具体采样规则，这些值在配置中集中列出。
- **正则形式**：法线平滑使用邻域法线差；内部损失使用 `relu(f_inner(center))²`，尺度损失惩罚超过 `scale_target` 的尺度。另加 `lambda_nesting` 控制的内外嵌套损失，防止独立内场穿出外场；这是实现补充，设为 0 可移除，不是论文式 16 的原项。
- **边界和 SSIM**：RGB 排除掠射；第三阶段从实际图像采连续 patch，额外排除壳命中边界附近像素。SSIM 采用 11×11 高斯窗，并排除触及无效像素的窗口。前两阶段采用随机射线与法线夹角阈值，不计算完整图像轮廓带。
- **高斯深度与增密**：按射线上 Mahalanobis 最近点排序。增密统计采用世界坐标位置梯度，保留 Adam 状态；克隆分摊密度，分裂按体积修正密度。其阈值不能直接视作原 3DGS 的屏幕空间梯度阈值。
- **加速实现**：使用 PyTorch 分块积分与实时刷新的协方差包围盒筛选，没有实现论文未公开细节的高性能 CUDA 加速结构。统一排序仍需与 `ray_chunk × 候选高斯数` 成正比的内存；不主张达到论文速度。
- **复杂路径**：只累加首外表面反射。内部全反射、交界顺序不正确、缺失交点等路径返回已知辐射项并标记 `valid=False`，不作为训练 RGB 监督；没有实现多次内反射。只穿过玻璃、不进入空腔的边缘射线使用两次外界面求交。几乎不透明的内部段可跳过后壁追踪。
- **颜色域**：默认把 8-bit sRGB 输入转为线性 RGB，环境保存线性 HDR；PNG 预览再转换到 sRGB 并截断。原始图像仍应含环境背景，RGBA 的 alpha 不参与几何监督。

## 数据与配置

支持以下已有姿态的数据：

```text
dataset/
  images/...
  sparse/0/cameras.bin  (或 .txt)
  sparse/0/images.bin   (或 .txt)
  sparse/0/points3D.bin (或 .txt)
```

COLMAP 相机须已去畸变，为 `PINHOLE` 或 `SIMPLE_PINHOLE`。保留完整主点位置，不假设主点在图像中心。

也支持 Blender/NeRF 的 `transforms_train.json` 和可选 `transforms_test.json`，或单个 `transforms.json`，其中 `transform_matrix` 采用 OpenGL/Blender camera-to-world 约定。相机可给出 `fl_x/fl_y/cx/cy`，或水平视场角 `camera_angle_x`。优先使用显式测试集；否则每五张留一张作为测试视图，不自动执行测试集评估。

所有坐标统一转换为 `x_normalized = (x_world - scene_center) / scene_scale`。相机位置和初始化点同时转换；方向和相机旋转保持不变。长度、尺度、SDF 都以归一化场景单位表示，吸收/密度为该长度的倒数。

配置模板为 `configs/gshell.json`，全部可选字段及默认值在 `gshell/config.py`。未添加新生产依赖；新增路径使用上游已用到的 PyTorch、NumPy、Pillow、plyfile，不依赖 tiny-cuda-nn，不要求编译上游 CUDA 光栅器来导入 GShell 路径。

## 使用入口

以下命令是使用说明，本次未执行：

```powershell
python train_gshell.py -s <dataset> -m output/gshell --config configs/gshell.json
python train_gshell.py -s <dataset> -m output/gshell --resume output/gshell/checkpoints/latest.pt
python render_gshell.py --checkpoint output/gshell/checkpoints/stage3.pt --output output/gshell/renders --split test
python export_gshell.py --checkpoint output/gshell/checkpoints/stage3.pt --output output/gshell/assets --resolution 96
```

断点保存模型、当前阶段和步数、优化器、随机数状态及配置。恢复使用断点配置；`--device` 可改运行设备。环境分辨率切换时重建该阶段的优化器；进入新阶段时使用独立优化器。

渲染每个视图输出 PNG 预览、线性 RGB `.npy` 和几何/路径有效性 `.png`。渲染脚本也可指定 `--source` 为另一套同坐标系姿态。`--shell-only`、`--interior-only` 分别隔离组件；`--environment another_environment.npy` 替换线性 `[H,W,3]` 环境。环境纬度轴为 +Z，经度 `atan2(y,x)`，不进行自动曝光匹配。

导出包含独立 `outer_shell.ply`、`inner_shell.ply`、`interior.ply`、`environment.npy` 与 `scene.json`。壳体网格采用 marching tetrahedra 提取并朝 SDF 正侧定向。PLY 内部高斯额外保存真实密度；其中为普通查看器提供的 opacity 只是参考光线近似，完整折射效果须由本渲染器和 checkpoint 重建。

## 验证范围

执行了新增 Python 文件的静态语法解析、配置字段检查和 Git 差异格式检查。没有安装训练环境，没有运行训练、渲染、数学测试或实验。

`tests/test_gshell_math.py` 提供解析积分、光学边界、隐式求交梯度和增密状态的数学测试源码，可在已有 PyTorch 环境中自行执行：

```powershell
python -m unittest discover -s tests -p "test_gshell_math.py"
```
