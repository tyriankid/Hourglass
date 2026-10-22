// 物理内核共用的结构体和常量。所有计算都在沙漏自身坐标系里：瓶颈在原点，轴向是 y。

struct SimU {
  g_dt: vec4f,      // 有效重力（含平移惯性力）.xyz，子步 dt
  omega_rc: vec4f,  // 角速度.xyz，颗粒团标称半径
  alpha_mc: vec4f,  // 角加速度.xyz，颗粒团标称质量（mg）
  gmin_cs: vec4f,   // 网格原点.xyz，网格边长
  gdim: vec4u,      // nx, ny, nz, 颗粒团容量
  drain: vec4f,     // 本子步孔口抽走目标（mg），上游方向（+1 上泡流、-1 下泡流、0 不流），单粒沙质量（mg），捕获半径
  fric: vec4f,      // 颗粒团静摩擦 μs，动摩擦 μk，与玻璃的摩擦，休眠速度
  misc: vec4f,      // shock propagation 系数，孔口出口速度，沙粒半径，SOR 系数
  ints: vec4u,      // 沙粒容量，是否按接触数平均，全部唤醒（沙漏在动），0
  lim: vec4f,       // 最大速度，沙粒停留速度，是否强制生成网格里攒的沙，速度阻尼系数
}

// 玻璃轮廓采样表：0..128 为 vec4(rIn, drIn, rOut, drOut)，129 为 vec4(H, 孔半径, 最大内半径, 端盖半径)
struct Profile { s: array<vec4f, 130> }

// 沙量操作（凝聚、消散、直方图、放置）的参数
struct OpU {
  a: vec4f,   // 凝聚：y0, y1（局部 y，带符号）, 泡（±1）, 数量；消散：阈值, 0, 泡, 0；放置：0, 0, 0, 数量
  b: vec4f,   // 随机种子, 时间（s）, 是否发光/出微光, 每个颗粒团的质量（mg）
  up: vec4f,  // 局部"向上"方向, 直方图下界
  r: vec4f,   // 直方图上界, 微光容量, 0, 0
}

const NONE: u32 = 0xffffffffu;
const CAP_MAX: u32 = 64u;
const SLOTS: u32 = 8u;

// ctl 缓冲的字下标
const C_FREE_TOP: u32 = 0u;     // 空闲颗粒团栈顶
const C_CAP_COUNT: u32 = 1u;    // 本子步捕获区里的颗粒团数
const C_GRAIN_HEAD: u32 = 3u;   // 沙粒环形缓冲写指针
const C_EMIT_COUNT: u32 = 4u;   // 本子步发射的沙粒数
const C_EMIT_BASE: u32 = 5u;    // 本子步发射起点
const C_EMIT_BUDGET: u32 = 6u;  // 未发射的小数粒（f32）
const C_RNG: u32 = 7u;          // 子步计数，用作随机种子
const C_EMIT_DIR: u32 = 20u;    // 最近一次流动的方向（f32，+1 往 -y 射出）
const C_FX_HEAD: u32 = 21u;     // 微光粒子环形缓冲写指针
// 以下也是每帧清零：两个泡里颗粒团的总质量（µg，定点）、本帧消散掉的质量（µg）
const C_MASS_POS: u32 = 24u;    // y > 0 的泡
const C_MASS_NEG: u32 = 25u;    // y < 0 的泡
const C_DISSOLVED: u32 = 26u;
const C_PLACED_FAIL: u32 = 27u;
const C_GRID_OVERFLOW: u32 = 28u; // 网格槽位不够、漏登记的颗粒团次数
const C_MOVING: u32 = 29u;      // 本帧各子步里速度 > 2 cm/s 的颗粒团计数之和
const C_CONDENSED: u32 = 30u;   // 本帧凝聚生成的质量（µg）
// 以下为每帧统计，CPU 在帧开始清零
const C_DRAINED: u32 = 8u;      // 本帧流过孔口的沙（mg，f32）
const C_EMITTED: u32 = 9u;      // 本帧发射的沙粒数
const C_IMPACT_SAND: u32 = 10u; // 沙粒撞沙堆次数
const C_IMPACT_GLASS: u32 = 11u;// 沙粒撞玻璃次数
const C_IMPACT_SPEED: u32 = 12u;// 撞击速度和（mm/s）
const C_MAX_SPEED: u32 = 13u;   // 颗粒团最大速度（0.1 mm/s）
const C_SPAWN_FAIL: u32 = 14u;  // 颗粒团池用尽次数（应恒为 0）
const C_DEPOSITED: u32 = 15u;   // 本帧沉积的沙粒数
const C_AVAIL: u32 = 16u;       // 捕获区里的沙量（mg，f32）
const C_CAP_OVERFLOW: u32 = 18u;// 捕获区溢出次数
const C_FORCED: u32 = 19u;      // 沙粒槽被复用时强制沉积的次数
const C_MERGED: u32 = 17u;      // 本帧合并掉的未满颗粒团数
const C_CAP_LIST: u32 = 32u;    // 捕获列表：64 × (下标, 距离)
const CTL_WORDS: u32 = 256u;
