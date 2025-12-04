import torch
from diffusers import ZImagePipeline

# 1. Load the pipeline
pipe = ZImagePipeline.from_pretrained(
    "/app/cold1/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# 2. Define 10 diverse prompts (you can edit these freely)
prompts = [
     "A vibrant spring garden bursting with cherry blossoms, tulips, and daffodils in full bloom, soft-focus background with sunlight filtering through trees, dreamy bokeh atmosphere, watercolor and ink illustration style, warm pastels palette, centered composition, no text, no letters, no words, no typography, no labels",
    "Cyberpunk cityscape at night with neon-lit skyscrapers, hovering vehicles leaving light trails, rain-slicked streets reflecting vivid magenta and cyan glows, photorealistic 3D render, high-contrast duotone lighting, dynamic diagonal flow, no text, no letters, no words, no typography, no labels",
    "Underwater ocean scene with translucent jellyfish pulsing with bioluminescence, schools of iridescent fish weaving through coral towers, shafts of sunlight piercing the turquoise depths, ethereal ambient lighting, digital painting style, monochrome teal with subtle gold accents, vertical symmetry, no text, no letters, no words, no typography, no labels",
    "Festival celebration in a mountain village at dusk: floating paper lanterns ascending into indigo sky, silhouetted dancers in flowing robes mid-motion, warm bonfire glow casting long shadows, hand-drawn sketch with soft gouache wash, earthy ochre and deep violet palette, wide panoramic composition, no text, no letters, no words, no typography, no labels",
    "Minimalist space exploration poster: a lone astronaut floating beside a geometric space station orbiting a ringed exoplanet, stars densely scattered in deep indigo void, clean vector flat design, cool duotone of slate gray and pale gold, centered composition with radial balance, no text, no letters, no words, no typography, no labels",
    "Autumn forest canopy viewed from below: overlapping maple and ginkgo leaves in fiery red, burnt orange, and golden yellow, sunlight flaring through gaps, impressionistic oil painting texture, rich warm palette, layered depth with foreground bokeh, no text, no letters, no words, no typography, no labels",
    "Surreal desert dreamscape with giant floating crystal formations refracting rainbow light, dunes sculpted into smooth wave-like ridges, distant mirage of arches, matte painting style, vibrant duotone of coral pink and dusty lavender, horizontal tripartite composition, no text, no letters, no words, no typography, no labels",
    "Alpine meadow at dawn: wildflowers carpeting rolling hills beneath snow-capped peaks, mist curling around pine clusters, soft glowing light, detailed pencil sketch with color pencil overlay, fresh pastel greens and sky blues, serene centered framing, no text, no letters, no words, no typography, no labels",
    "Futuristic botanical lab interior: transparent growth domes housing glowing flora, suspended root systems and vine networks pulsing with soft light, clean architecture with curved glass and brushed metal, isometric 3D render, biophilic palette of jade green and warm white, balanced asymmetrical layout, no text, no letters, no words, no typography, no labels",
    "Northern lights over frozen lake: aurora borealis dancing in emerald and violet ribbons above mirror-like ice, silhouettes of distant evergreens, starry sky with Milky Way arc, long-exposure photography aesthetic, cool monochrome with electric accents, expansive wide-angle composition, no text, no letters, no words, no typography, no labels",
    "Mystical library interior with towering shelves curving into a spiral dome, floating books emitting soft glows, shafts of golden light from stained-glass skylights, warm chiaroscuro lighting, detailed digital matte painting, rich amber and deep burgundy palette, centralized vanishing point perspective, no text, no letters, no words, no typography, no labels",
    "Volcanic coastline at twilight: molten lava cascading into ocean with dramatic steam plumes, basalt columns framing the scene, dynamic contrast of crimson and obsidian, hyperrealistic photography style, high-dynamic-range lighting, diagonal motion-driven composition, no text, no letters, no words, no typography, no labels",
    "Steampunk airship fleet sailing through cotton-candy clouds above mountain peaks, brass propellers spinning, intricate pipe and gear detailing, sepia-toned vector illustration with copper highlights, vintage duotone palette, layered depth with foreground airship in focus, no text, no letters, no words, no typography, no labels",
    "Zen rock garden viewed from above: raked gravel swirls surrounding moss-covered boulders and a single twisted pine, soft shadow gradients at mid-morning, minimalist ink wash painting style, monochrome gray with subtle sage green accents, balanced asymmetry, no text, no letters, no words, no typography, no labels",
    "Bioluminescent jungle at night: giant glowing mushrooms, ferns with pulsating veins, vines dripping liquid light, firefly-like orbs drifting slowly, cinematic 3D render, cool teal and violet palette with amber focal points, immersive wide-angle perspective, no text, no letters, no words, no typography, no labels",
    "Arctic expedition scene: sled dogs racing across wind-sculpted ice dunes under pale sun, distant auroral haze, frost-laden fur details, documentary-style realism, icy blue and charcoal monochrome, strong horizontal banding with dynamic motion blur, no text, no letters, no words, no typography, no labels",
    "Floating islands in sky archipelago: waterfalls cascading into clouds, suspended temples and bridges connecting landmasses, soft volumetric lighting, Studio Ghibli-inspired hand-painted aesthetic, airy pastel palette of sky blue, lavender, and cream, dreamy centered composition, no text, no letters, no words, no typography, no labels",
    "Retro-futuristic observatory on lunar surface: domed telescopes with analog dials, Earth rising over cratered horizon, vintage sci-fi poster style, warm duotone of mustard yellow and slate gray, bold geometric framing, no text, no letters, no words, no typography, no labels",
    "Coral reef metropolis: architecture grown from living coral and seashells, schools of fish flowing through archways, sunlight refracting in turquoise water, isometric 3D illustration, vibrant tropical palette, clean vector aesthetic with organic textures, no text, no letters, no words, no typography, no labels",
    "Autumn harvest field at golden hour: windrows of wheat forming concentric patterns, lone scarecrow with flowing fabric, backlit by low sun, impressionist brushwork, warm amber and burnt sienna tones, radial composition drawing eye to horizon, no text, no letters, no words, no typography, no labels",
    "Deep-space nebula core: swirling gas clouds in magenta and cyan, newborn stars igniting in clusters, gravitational lensing distortions, Hubble-inspired astrophotography enhanced with painterly detail, cosmic duotone palette, immersive fisheye perspective, no text, no letters, no words, no typography, no labels",
    "Bamboo forest after rain: droplets suspended mid-air, light beams piercing through tall stalks, mist hugging the ground, serene monochrome ink drawing with subtle celadon wash, vertical rhythm and repetition, no text, no letters, no words, no typography, no labels",
    "Desert oasis mirage sequence: layered transparent reflections of palm groves and water over dunes, heat haze distortion, surreal photomontage style, warm ochre and aqua duotone, symmetrical mirroring effect, no text, no letters, no words, no typography, no labels",
    "Underground crystal cavern: massive geodes glowing from within, subterranean river reflecting prismatic light, stalactites forming rhythmic patterns, fantasy 3D render, jewel-toned palette of amethyst, sapphire, and citrine, cavern entrance framing the view, no text, no letters, no words, no typography, no labels",
    "Kite festival on coastal cliffs: hundreds of colorful diamond kites soaring in synchronized arcs against sea and sky, dynamic wind lines implied by fabric tension, vibrant vector flat design, bold primary triad palette, high-energy diagonal flow, no text, no letters, no words, no typography, no labels",
    "Antarctic iceberg interior: glowing blue ice tunnels with natural arches and chambers, light filtering through translucent walls, macro ice texture detail, cinematic wide shot, cool monochrome with glowing cyan core, serene minimal framing, no text, no letters, no words, no typography, no labels",
    "Mechanical forest: trees with copper trunks and gear-driven branches, blossoms made of delicate clockwork petals, steam rising from root vents, intricate steampunk cross-section illustration, warm brass and moss green palette, centered organic-tech hybrid composition, no text, no letters, no words, no typography, no labels",
    "Tidal rock pools at sunset: interconnected basins reflecting fiery sky, sea anemones and starfish in vivid contrast, water surface like liquid gold, realistic macro photography style, warm duotone of coral and slate, intimate close-up framing, no text, no letters, no words, no typography, no labels",
    "Floating market on jungle river: wooden canoes laden with fruits and flowers, thatched huts on stilts among mangroves, morning mist and dappled light, painterly gouache texture, lush emerald and terracotta palette, meandering S-curve composition, no text, no letters, no words, no typography, no labels",
    "Zero-gravity dance performance in orbital station: performers in flowing garments suspended mid-leap, ribbons tracing elegant trajectories, soft ambient lighting from ring modules, elegant 3D render with motion blur, elegant duotone of pearl white and deep navy, central vortex composition, no text, no letters, no words, no typography, no labels",
    "春日山野梯田全景：层叠水田如镜面倒映粉白云霞，农人牵牛缓行田埂，远山淡影朦胧，水墨淡彩风格，青绿与浅赭主调，横向延展构图，画面宁静悠远，无文字，无字母，无单词，无排版，无标签",
    "未来生态城市空中花园：悬浮绿岛串联透明廊桥，垂直森林与流线型建筑共生，飞鸟与滑翔器穿梭其间，柔和日光漫射，清新插画风格，薄荷绿与暖灰搭配，鸟瞰对称布局，无文字，无字母，无单词，无排版，无标签",
    "敦煌风沙幻境：起伏沙丘形成天然螺旋纹路，远古岩画轮廓若隐若现于崖壁，月光洒落银辉，矿物颜料质感厚涂风格，土红、石青、金箔三色主调，纵深透视构图，无文字，无字母，无单词，无排版，无标签",
    "江南水乡雨巷：青石板路泛着水光，乌篷船静泊拱桥下，纸伞斜倚门边，细雨如丝，朦胧水彩晕染风格，黛青与烟灰主色，竖向纵深构图，诗意留白，无文字，无字母，无单词，无排版，无标签",
    "深海热泉生态系统：黑烟囱喷涌矿物流体，巨型管蠕虫摇曳如花，发光虾群环绕升腾，蓝紫冷光主导，写实3D渲染，深邃幽暗氛围，中心聚焦式构图，无文字，无字母，无单词，无排版，无标签",
    "川西秋林秘境：层林尽染金红橙黄，溪流蜿蜒穿行石滩，薄雾缭绕林间，细腻胶片摄影质感，浓郁暖色调，S形引导线构图，无文字，无字母，无单词，无排版，无标签",
    "苗岭晨曦梯田：晨雾如纱漫过层层稻浪，吊脚楼群隐现山腰，炊烟袅袅升腾，工笔重彩融合现代插画，靛蓝、银灰与朱砂点缀，平衡式全景构图，无文字，无字母，无单词，无排版，无标签",
    "冰雪极光穹顶：冰晶穹顶下仰望舞动绿紫极光，冰面倒影星轨旋转，极简留白空间，清冷数字绘画风格，冰蓝、银白与幽绿主调，圆形聚焦构图，无文字，无字母，无单词，无排版，无标签",
    "西南喀斯特峰林云海：锥状山峰破云而出，溶洞暗河隐现幽光，苍鹰盘旋峰顶，中国山水画意境结合数字渲染，墨色氤氲配淡金晨曦，高远法三段式构图，无文字，无字母，无单词，无排版，无标签",
    "敦煌藻井结构演化图景：中心莲花层层展开为几何星芒，飞天衣带化作流线藤蔓，矿物色渐变过渡，装饰性矢量平面风格，石青、土红、金三色经典配比，绝对中心对称构图，无文字，无字母，无单词，无排版，无标签",
    "江南蚕乡春事：桑园新叶滴翠，竹匾铺晒雪白蚕茧，木架上悬垂银丝缕缕，柔光漫射，温润水彩与钢笔线描结合，嫩绿、米白、浅褐清新配色，横向叙事性构图，无文字，无字母，无单词，无排版，无标签",
    "秦岭雪松雾凇奇观：松枝裹满晶莹冰甲，雪压枝低形成天然拱门，林间光柱斜射，高清微距摄影质感，纯白、冰蓝与松针墨绿对比，框架式前景构图，无文字，无字母，无单词，无排版，无标签",
    "闽南红砖古厝群落：燕尾脊错落起伏，天井中三角梅盛放，斑驳砖墙藤蔓攀援，暖调胶片纪实风格，朱红、灰白与苔绿主色，俯角散点透视，无文字，无字母，无单词，无排版，无标签",
    "高原盐湖镜像世界：天空云影完整倒映于浅水盐池，结晶盐花如绽放冰晶，无人迹空旷感，极简主义摄影风格，粉紫、钴蓝与纯白构成梦幻色调，绝对水平对称构图，无文字，无字母，无单词，无排版，无标签",
    "徽州秋晒图景：黛瓦白墙间高架竹匾铺满金黄玉米、火红辣椒，远山层叠如屏，温暖柔焦影像风格，浓郁丰收色谱，点线面节奏布局，无文字，无字母，无单词，无排版，无标签",
    "丝绸之路上的风蚀雅丹：巨岩如舰队列阵，风痕刻出平行波纹，孤驼剪影行于沙脊，超现实主义油画笔触，赭石、沙金与靛蓝天幕，低视角强化体量感，无文字，无字母，无单词，无排版，无标签",
    "岭南骑楼雨季：满洲窗透出柔光，雨水沿弧形廊柱垂落成帘，青石路反光映出霓虹倒影，复古胶片颗粒感，深绿、砖红与暖黄主调，纵深廊道引导线构图，无文字，无字母，无单词，无排版，无标签",
    "藏地经幡山谷：五彩经幡横跨峡谷随风翻飞，雪山背景肃穆，苍鹰掠过旗阵，厚重肌理丙烯画风，藏青、朱红、明黄、白、绿五色体系，横向延展动态构图，无文字，无字母，无单词，无排版，无标签",
    "江南茶山云雾采撷时：茶垄如绿浪起伏，采茶人斗笠身影隐现雾中，竹篓半满新芽，清雅水墨淡设色风格，茶绿、灰白、浅褐为主，散点透视营造空间层次，无文字，无字母，无单词，无排版，无标签",
    "敦煌月牙泉夜色：沙山环抱一弯碧水，倒映银河与星轨，骆驼队剪影缓行泉畔，长曝光星空摄影融合工笔细节，深蓝夜空配沙金与水银色，圆形泉眼居中构图，无文字，无字母，无单词，无排版，无标签",
     "热带雨林树冠层俯视图：层层叠叠的巨叶形成绿色迷宫，附生凤梨盛满雨水，树懒蜷于藤蔓间，阳光穿透形成光柱阵列，生物多样性插画风格，翡翠绿与琥珀金交织，蜂窝状六边形构图，无文字，无字母，无单词，无排版，无标签",
    "废弃剧院内部：破败天鹅绒幕布半垂，水晶吊灯覆满蛛网仍在微光闪烁，座椅缝隙钻出野花，柔焦超现实摄影，灰紫与锈红主调，纵深舞台透视构图，无文字，无字母，无单词，无排版，无标签",
    "火山熔岩冷却地貌：玄武岩六棱柱群如管风琴矗立，蒸汽从裂缝袅袅升腾，硫磺结晶点缀岩表，地质写实渲染，炭黑、赭红与硫磺黄对比，仰视强化垂直张力，无文字，无字母，无单词，无排版，无标签",
    "鲸落生态系统剖面：巨型鲸骨沉于深海泥床，盲虾与铠甲虾攀附骨殖，化能合成菌毯如雪覆盖，冷光生物发光点缀，科学插画精度，暗蓝底色配荧光橙局部，垂直分层构图，无文字，无字母，无单词，无排版，无标签",
    "高原湿地晨雾：浅水沼泽如镜，黑颈鹤单腿独立倒影清晰，芦苇丛染金边，柔光薄雾弥散，东方水墨留白意境，灰蓝、银白与淡赭过渡，横向低视角构图，无文字，无字母，无单词，无排版，无标签",
    "未来垂直农场中庭：螺旋种植塔环绕中央光井，作物层叠如彩虹色带，机器人臂轻巧巡检，洁净科技插画风格，生机绿与钛白主调，旋转上升动线构图，无文字，无字母，无单词，无排版，无标签",
    "喀斯特地下河溶洞：钟乳石与石笋几近接合，地下河泛着幽蓝磷光，盲鱼群游弋无声，探险灯束切开黑暗，神秘暗调渲染，靛蓝与钙白对比，隧道式透视构图，无文字，无字母，无单词，无排版，无标签",
    "沙漠绿洲鸟瞰：棕榈树冠呈同心圆分布，水道如银线蜿蜒，沙丘波纹如凝固海浪，卫星影像美学融合手绘质感，沙金、翠绿与天青主色，几何中心点构图，无文字，无字母，无单词，无排版，无标签",
    "极地冰裂隙内部：冰层透出千年蓝光，裂缝深不见底，冰晶结构如哥特教堂拱顶，冷冽高清微距摄影，冰蓝渐变至深空紫，垂直纵深构图，无文字，无字母，无单词，无排版，无标签",
    "古蜀三星堆神树意象重构：青铜神树枝干分九杈，龙形饰件盘绕升腾，云雷纹背景若隐若现，金属蚀刻与数字浮雕结合，青铜绿锈与暗金光泽，中心放射状构图，无文字，无字母，无单词，无排版，无标签",
    "潮间带退潮时分：潮池如星罗棋布镶嵌礁石，海葵舒展触手，寄居蟹背负彩壳穿行，湿润反光表面，生态纪实摄影风格，灰褐基底配珊瑚粉与海葵橙，密集点阵构图，无文字，无字母，无单词，无排版，无标签",
    "废弃太空站外部：太阳能板半毁飘浮，藤壶状外星附着生物覆盖舱体，地球弧线悬于背景，科幻废土写实风格，铁锈红与太空黑主调，失重散点构图，无文字，无字母，无单词，无排版，无标签",
    "苗族银饰锻造工坊一角：火塘微光映照锤痕累累的砧台，半成银片悬于架上，木模与蜡雕静待使用，暖调纪实摄影，铜红、银白与松烟黑，紧凑特写式构图，无文字，无字母，无单词，无排版，无标签",
    "云贵高原梯田灌水期：水田如万面银镜拼贴山体，倒映流云飞鸟，田埂线条如书法飞白，航拍摄影结合水墨晕染，银白、黛青与浅灰主调，抽象几何构图，无文字，无字母，无单词，无排版，无标签",
    "深海热液喷口生物群：巨型管蠕虫红冠摇曳，雪人蟹挥螯巡游，硫化物烟囱喷涌黑烟，冷暖光对比强烈，科学幻想插画风格，血红、硫黄与深海黑，中心喷口聚焦构图，无文字，无字母，无单词，无排版，无标签",
    "江南丝织工坊织机特写：木机齿轮咬合精密，素色经线与彩纬交织未完成锦缎，窗外竹影摇曳，温润木色摄影风格，原木褐、蚕丝白与茜草红点缀，框架式局部构图，无文字，无字母，无单词，无排版，无标签",
    "阿尔卑斯高山草甸盛夏：雪绒花与龙胆星布绿茵，岩羊跃过碎石坡，远峰终年积雪，清新自然主义绘画，钴蓝、草绿与雪白主色，开阔全景构图，无文字，无字母，无单词，无排版，无标签",
    "敦煌莫高窟第220窟藻井数字化复原：团花卷草纹层层叠晕，飞天反弹琵琶姿态凝固，矿物颜料饱和如初，数字文物修复高清呈现，青金石蓝、朱砂红、金箔三重阶，绝对中心对称，无文字，无字母，无单词，无排版，无标签",
    "闽南土楼航拍晨景：圆形夯土巨构嵌入梯田，炊烟从中央天井袅袅升起，门坪晾晒稻谷呈扇形，大地艺术摄影风格，土黄、稻金与瓦灰主调，同心圆构图，无文字，无字母，无单词，无排版，无标签",
    "深海鲸歌声波可视化意象：声波如金色涟漪扩散于墨蓝水体，浮游生物随频率聚散成图腾，抽象数据艺术风格，金、靛、黑三色流动，同心扩散构图，无文字，无字母，无单词，无排版，无标签",
    "藏地转山道风马旗阵：经幡长阵沿山脊绵延数里，风过时如彩浪翻涌，雪山背景肃穆，广角动态摄影，五彩主调融入苍穹蓝，线性引导构图，无文字，无字母，无单词，无排版，无标签",
    "岭南镬耳屋群雨后：青瓦屋顶积水如镜，倒映云影与高耸镬耳山墙，青苔沿墙根蔓延，柔焦胶片质感，黛青、灰白与苔绿主色，错落屋顶节奏构图，无文字，无字母，无单词，无排版，无标签",
    "未来沙漠水收集站：大型仿生集露装置如银色花朵绽放沙海，冷凝水沿导管汇入地下储池，极简可持续设计美学，钛银与沙金主调，单体中心构图，无文字，无字母，无单词，无排版，无标签",
    "川西藏寨石砌碉楼群：石片干砌墙面肌理粗犷，窗棂彩绘吉祥八宝纹样，经幡在风中猎猎，高原硬光摄影，赭石、藏青与白灰主色，聚落聚散构图，无文字，无字母，无单词，无排版，无标签",
    "热带珊瑚产卵夜：海水泛起粉红光雾，珊瑚同步释放精卵团如星群升腾，鱼群静默环绕，长曝光水下摄影，黑蓝底色配荧光粉橙爆点，弥漫式构图，无文字，无字母，无单词，无排版，无标签",
    "徽州古桥雪霁：单拱石桥覆薄雪，桥洞与倒影成完美圆环，枯柳枝桠划破素绢，文人画淡墨风格，雪白、淡墨与赭石轻染，圆形留白构图，无文字，无字母，无单词，无排版，无标签",
    "废弃游乐园摩天轮：锈蚀钢架攀满常春藤，座舱半开悬于半空，秋阳斜照拉长阴影，废土诗意摄影，铁锈橙、枯叶黄与雾灰主调，对角线失衡构图，无文字，无字母，无单词，无排版，无标签",
    "长江源冰川融水溪流：清流穿行于冰碛砾石间，水底卵石清晰可见，远山雪线如银边，高清生态摄影，冰蓝、石青与沙金主色，Z字形水流引导构图，无文字，无字母，无单词，无排版，无标签",
    "苗岭百褶裙纹样灵感场景：梯田层叠如裙褶，晨雾流动似银饰反光，飞鸟掠过形成动态点缀，装饰性平面设计风格，靛蓝底配银线几何，垂直节奏构图，无文字，无字母，无单词，无排版，无标签",
    "深海发光水母群舞：伞盖透明如琉璃，触手垂落光丝，群体聚散成螺旋阵列，暗场微光摄影，深黑底色配青白冷光，螺旋上升构图，无文字，无字母，无单词，无排版，无标签",
    "敦煌月牙泉鸣沙山光影：沙脊曲线如凝固波浪，光影分割明暗几何，一弯碧水静卧环抱，抽象风景摄影，金、赭、青三色构成，极简负空间构图，无文字，无字母，无单词，无排版，无标签",
    "江南油纸伞作坊晾晒场：百把伞面半开如花海，桐油反光微亮，竹骨纤细有序，柔光平拍风格，朱红、靛蓝、竹青主调，重复阵列构图，无文字，无字母，无单词，无排版，无标签",
    "青藏铁路沿线藏羚羊迁徙：列车静默停靠观景台，羊群如银线横穿冻土带，远山雪峰连绵，纪实宏大摄影，藏青、银灰与雪白主色，水平带状构图，无文字，无字母，无单词，无排版，无标签",
    "喀斯特峰林溶洞出口：光束从洞口倾泻而下，照亮钟乳石森林，雾气氤氲如仙境，电影级光影渲染，金光与幽蓝对比，明暗交界线构图，无文字，无字母，无单词，无排版，无标签",
    "闽南红砖厝砖雕细节：墙面嵌砌花鸟人物砖雕，风化痕迹与新生苔藓共存，侧光凸显浮雕层次，文物特写摄影，砖红、灰白与翠绿点缀，微距局部构图，无文字，无字母，无单词，无排版，无标签",
    "未来城市垂直湿地：建筑立面覆盖水生植物，净化水流沿生态墙层叠跌落，蜻蜓停驻芦苇，生态科技插画，生机绿与清水蓝主调，垂直分层构图，无文字，无字母，无单词，无排版，无标签",
    "川西林盘院落秋色：竹林环抱青瓦房，银杏叶铺满石板院，陶缸盛满雨水映天，温润乡土摄影，金黄、黛瓦灰与竹青主色，围合式庭院构图，无文字，无字母，无单词，无排版，无标签",
    "深海热泉‘黑烟囱’喷发瞬间：超高温流体与海水激撞形成矿物烟柱，耐热微生物云团环绕，高速摄影冻结动态，暗黑底色配硫黄黄与铁锈红，中心爆发式构图，无文字，无字母，无单词，无排版，无标签",
    "藏地玛尼堆朝圣路：石堆层层叠叠绵延山脊，彩色牦牛毛绳缠绕石间，风蚀经文石静默，高原广角摄影，赭石、白石与经幡彩条，线性延伸构图，无文字，无字母，无单词，无排版，无标签",
    "岭南满洲窗光影实验：彩色玻璃拼花投射光斑于素墙，光谱随日移变幻，极简构成摄影，红、黄、蓝三原色光斑配白墙，几何光构图，无文字，无字母，无单词，无排版，无标签",
    "敦煌壁画飞天动态解构：多帧姿态重叠如频闪摄影，飘带轨迹化作金色流线，矿物色背景渐变，数字动态艺术风格，石青底配金线轨迹，螺旋上升构图，无文字，无字母，无单词，无排版，无标签",
    "长江三峡纤夫道遗迹：青石台阶深陷绳痕，野草从缝隙钻出，江水奔流于崖下，历史感纪实摄影，青灰、苔绿与江水褐主色，斜线遗迹引导构图，无文字，无字母，无单词，无排版，无标签",
    "苗族蜡染作坊染缸阵列：靛蓝染缸大小错落，布匹半浸缸中泛起涟漪，木架悬挂待晒蓝布，沉静工艺摄影，靛蓝、木褐与水银灰主调，阵列重复构图，无文字，无字母，无单词，无排版，无标签",
    "青藏高原盐湖结晶微观：卤水蒸发形成几何盐晶簇，如微型水晶宫殿，偏振光摄影显色，粉紫、钴蓝与纯白构成，微距晶体对称构图，无文字，无字母，无单词，无排版，无标签",
    "徽州古村落晨炊：马头墙轮廓剪影，炊烟从天井袅袅升腾，薄雾漫过稻田，水墨氤氲风格，淡墨、灰白与赭石轻染，层叠纵深构图，无文字，无字母，无单词，无排版，无标签",
    "废弃核电站冷却塔内景：藤蔓从顶部破口垂落，雨滴沿弧形内壁滑落成线，地面积水倒映穹顶，废土崇高美学，混凝土灰与植物绿对比，仰视穹顶构图，无文字，无字母，无单词，无排版，无标签",
    "热带红树林根系网络：气根如高跷林立浅水，招潮蟹洞口星布泥滩，退潮后水镜倒影完整，生态航拍风格，橄榄绿、泥褐与水银灰主调，交织网状构图，无文字，无字母，无单词，无排版，无标签",
    "敦煌星图洞窟天顶：拟构古代天文图式，星宿连线成神话形象，青金石色夜空深邃，数字复原壁画风格，群青、金箔与铅白主色，圆形天穹构图，无文字，无字母，无单词，无排版，无标签",
    "江南蚕室暖阁：竹匾层叠架设，蚕群密布食桑，蒸汽氤氲窗纸，柔光纪实风格，竹青、蚕白与桑叶绿主调，层叠纵深构图，无文字，无字母，无单词，无排版，无标签",
    "川藏线怒江72拐航拍：盘山公路如银带缠绕山体，越野车小如甲虫，峡谷深邃云雾缭绕，壮阔地理摄影，赭石山体配银灰公路与翠绿谷底，螺旋下降构图，无文字，无字母，无单词，无排版，无标签"
]

# Optional: set base seed for reproducibility
base_seed = 200303031715

# 3. Generate one image per prompt
for i, prompt in enumerate(prompts, start=29):
    seed = base_seed + i  # unique seed per image
    generator = torch.Generator("cuda").manual_seed(seed)

    print(f"Generating image {i}/10: '{prompt[:50]}...'")
    
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]

    filename = f"example_{i:03d}.png"
    image.save(filename)
    print(f"✅ Saved: {filename}")