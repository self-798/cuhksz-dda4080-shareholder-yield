"""
HSICS Industry Mapping for HSCI Constituent Stocks
===================================================
Based on the official 恒生行业分类系统 (HSICS) classification.

Hierarchy: 12 Industries -> 31 Sectors -> 112 Sub-sectors
Codes: 00 Energy, 05 Materials, 10 Industrials, 23 Cons. Disc.,
       25 Cons. Staples, 28 Healthcare, 35 Telecom, 40 Utilities,
       50 Financials, 60 Properties, 70 IT, 80 Conglomerates

Mapping strategy (priority order):
  1. Verified akshare industry name -> HSICS code lookup
  2. Stock code range fallback (rough industry groupings)
"""

import json
import os

# ============================================================================
# SECTION 1: COMPLETE HSICS HIERARCHY (from official PDF)
# ============================================================================

# Industry code -> (Chinese name, English name)
HSICS_INDUSTRIES = {
    "00": ("能源业",         "Energy"),
    "05": ("原材料业",       "Materials"),
    "10": ("工业",           "Industrials"),
    "23": ("非必需性消费",   "Consumer Discretionary"),
    "25": ("必需性消费",     "Consumer Staples"),
    "28": ("医疗保健业",     "Healthcare"),
    "35": ("电讯业",         "Telecommunications"),
    "40": ("公用事业",       "Utilities"),
    "50": ("金融业",         "Financials"),
    "60": ("地产建筑业",     "Properties & Construction"),
    "70": ("资讯科技业",     "Information Technology"),
    "80": ("综合企业",       "Conglomerates"),
}

# Sector code -> (industry_code, sector_name)
HSICS_SECTORS = {
    "0010": ("00", "石油及天然气"),
    "0020": ("00", "煤炭"),
    "0510": ("05", "黄金及贵金属"),
    "0520": ("05", "一般金属及矿石"),
    "0530": ("05", "原材料"),
    "1010": ("10", "工业工程"),
    "1020": ("10", "工用运输"),
    "1030": ("10", "工用支援"),
    "2310": ("23", "汽车"),
    "2320": ("23", "家庭电器及用品"),
    "2330": ("23", "纺织及服饰"),
    "2340": ("23", "旅游及消闲设施"),
    "2350": ("23", "媒体及娱乐"),
    "2360": ("23", "支援服务"),
    "2370": ("23", "专业零售"),
    "2510": ("25", "食物饮品"),
    "2520": ("25", "农业产品"),
    "2530": ("25", "消费者主要零售商"),
    "2810": ("28", "药品及生物科技"),
    "2820": ("28", "其他医疗保健"),
    "3500": ("35", "电讯"),
    "4000": ("40", "公用事业"),
    "5010": ("50", "银行"),
    "5020": ("50", "保险"),
    "5030": ("50", "其他金融"),
    "6010": ("60", "地产"),
    "6020": ("60", "建筑"),
    "7010": ("70", "资讯科技器材"),
    "7020": ("70", "软件服务"),
    "7030": ("70", "半导体"),
    "8000": ("80", "综合企业"),
}

# All 112 sub-sectors with their parent industry codes
# Format: sub-sector name -> 2-digit industry code
HSICS_SUBSECTORS = {
    # 00 能源业 (Energy)
    "油气生产商":         "00",
    "油气设备与服务":     "00",
    "煤炭":               "00",

    # 05 原材料业 (Materials)
    "黄金及贵金属":       "05",
    "钢铁":               "05",
    "铜":                 "05",
    "铝":                 "05",
    "其他金属及矿物":     "05",
    "化肥及农用化合物":   "05",
    "林业及木材":         "05",
    "纸及纸制品":         "05",
    "特殊化工用品":       "05",

    # 10 工业 (Industrials)
    "商用运输工具及货车": "10",
    "轨道与列车设备":     "10",
    "工业零件及器材":     "10",
    "电子零件":           "10",
    "环保工程":           "10",
    "重型机械":           "10",
    "新能源物料":         "10",
    "航空航天与国防":     "10",
    "能源储存装置":       "10",
    "航运及港口":         "10",
    "铁路及公路":         "10",
    "航空货运及物流":     "10",
    "公路运输":           "10",
    "采购及供应链管理":   "10",
    "印刷及包装":         "10",

    # 23 非必需性消费 (Consumer Discretionary)
    "汽车":               "23",
    "汽车零件":           "23",
    "摩托车及其他":       "23",
    "家庭电器":           "23",
    "消费电子产品":       "23",
    "玩具及消闲用品":     "23",
    "家具":               "23",
    "宠物用品":           "23",
    "纺织品及布料":       "23",
    "服装":               "23",
    "鞋类":               "23",
    "珠宝钟表":           "23",
    "其他服饰配件":       "23",
    "公共运输":           "23",
    "航空服务":           "23",
    "赌场及博彩":         "23",
    "酒店及度假村":       "23",
    "旅游及观光":         "23",
    "餐饮":               "23",
    "消闲及文娱设施":     "23",
    "广告及宣传":         "23",
    "广播":               "23",
    "影视娱乐":           "23",
    "出版":               "23",
    "互动媒体及服务":     "23",
    "教育":               "23",
    "其他支援服务":       "23",
    "汽车零售商":         "23",
    "服装零售商":         "23",
    "家居装修零售商":     "23",
    "多元化零售商":       "23",
    "其他零售商":         "23",
    "线上零售商":         "23",

    # 25 必需性消费 (Consumer Staples)
    "包装食品":           "25",
    "乳制品":             "25",
    "非酒精饮料":         "25",
    "酒精饮料":           "25",
    "烟草":               "25",
    "食品添加剂":         "25",
    "禽畜肉类":           "25",
    "农产品":             "25",
    "禽畜饲料":           "25",
    "超市及便利店":       "25",
    "个人护理":           "25",
    "家居消耗品":         "25",

    # 28 医疗保健业 (Healthcare)
    "药品":               "28",
    "生物技术":           "28",
    "中医药":             "28",
    "药品分销":           "28",
    "医疗设备及用品":     "28",
    "医疗及医学美容服务": "28",
    "膳食补充品":         "28",
    "护肤与化妆品":       "28",

    # 35 电讯业 (Telecommunications)
    "卫星及无线通讯":     "35",
    "电讯服务":           "35",

    # 40 公用事业 (Utilities)
    "常规电力":           "40",
    "燃气供应":           "40",
    "水务":               "40",
    "非传统/可再生能源":  "40",
    "核能":               "40",

    # 50 金融业 (Financials)
    "银行":               "50",
    "保险":               "50",
    "证券及经纪":         "50",
    "投资及资产管理":     "50",
    "信贷":               "50",
    "其他金融":           "50",
    "支付服务":           "50",

    # 60 地产建筑业 (Properties & Construction)
    "地产代理":           "60",
    "地产发展商":         "60",
    "地产投资":           "60",
    "房地产投资信托":     "60",
    "物业服务及管理":     "60",
    "建筑材料":           "60",
    "楼宇建造":           "60",
    "重型基建":           "60",

    # 70 资讯科技业 (Information Technology)
    "消费性电讯设备":     "70",
    "电讯网路基建设施":   "70",
    "电脑及周边器材":     "70",
    "数码解决方案服务":   "70",
    "互联网服务及基础设施":"70",
    "应用软件":           "70",
    "游戏软件":           "70",
    "半导体":             "70",
    "半导体设备与材料":   "70",

    # 80 综合企业 (Conglomerates)
    "综合企业":           "80",
}

# ============================================================================
# SECTION 2: EXTENDED NAME MAPPINGS (Sector-level + common aliases)
# ============================================================================

# Sector-level Chinese names -> industry code
SECTOR_TO_INDUSTRY = {
    sector_name: industry_code
    for sector_code, (industry_code, sector_name) in HSICS_SECTORS.items()
}

# Additional common aliases / shortened names -> industry code
NAME_ALIASES = {
    # Energy
    "石油及天然气": "00",
    "能源":         "00",

    # Materials
    "一般金属及矿石": "05",
    "原材料":         "05",
    "化工":           "05",
    "金属":           "05",
    "矿业":           "05",

    # Industrials
    "工业工程":       "10",
    "工用运输":       "10",
    "工用支援":       "10",
    "运输":           "10",
    "物流":           "10",
    "航运":           "10",
    "航空货运":       "10",
    "铁路":           "10",
    "公路":           "10",
    "港口":           "10",
    "建筑":           "10",
    "基建":           "10",
    "工程":           "10",
    "制造业":         "10",
    "机械":           "10",
    "航天":           "10",
    "国防":           "10",
    "电池":           "10",
    "储能":           "10",
    "包装":           "10",
    "印刷":           "10",

    # Consumer Discretionary
    "非必需消费":     "23",
    "消费":           "23",
    "汽车":           "23",
    "家电":           "23",
    "家庭电器及用品": "23",
    "家庭用品":       "23",
    "纺织":           "23",
    "服饰":           "23",
    "纺织及服饰":     "23",
    "旅游及消闲设施": "23",
    "旅游":           "23",
    "酒店":           "23",
    "博彩":           "23",
    "消闲":           "23",
    "娱乐":           "23",
    "媒体":           "23",
    "媒体及娱乐":     "23",
    "广告":           "23",
    "影视":           "23",
    "零售":           "23",
    "专业零售":       "23",
    "支援服务":       "23",

    # Consumer Staples
    "必需消费":       "25",
    "消费者主要零售商":"25",
    "食物饮品":       "25",
    "食品":           "25",
    "饮料":           "25",
    "饮品":           "25",
    "乳品":           "25",
    "酒类":           "25",
    "农业":           "25",
    "农业产品":       "25",
    "超市":           "25",
    "便利店":         "25",

    # Healthcare
    "医疗":           "28",
    "医药":           "28",
    "医疗保健":       "28",
    "药品及生物科技": "28",
    "生物科技":       "28",
    "其他医疗保健":   "28",
    "医疗设备":       "28",
    "美容":           "28",

    # Telecommunications
    "电讯":           "35",
    "通讯":           "35",
    "电信":           "35",

    # Utilities
    "公用事业":       "40",
    "电力":           "40",
    "燃气":           "40",
    "供水":           "40",
    "可再生能源":     "40",
    "新能源":         "40",

    # Financials
    "金融":           "50",
    "证券":           "50",
    "券商":           "50",
    "基金":           "50",
    "资产管理":       "50",
    "投资":           "50",
    "经纪":           "50",

    # Properties & Construction
    "地产":           "60",
    "房地产":         "60",
    "物业":           "60",
    "建造":           "60",
    "建材":           "60",
    "REIT":           "60",
    "地产建筑业":     "60",

    # IT
    "科技":           "70",
    "信息科技":       "70",
    "资讯科技":       "70",
    "资讯科技器材":   "70",
    "软件":           "70",
    "软件服务":       "70",
    "互联网":         "70",
    "游戏":           "70",
    "电讯设备":       "70",
    "数码":           "70",
    "数字":           "70",
    "云端":           "70",
    "数据中心":       "70",
    "芯片":           "70",
    "电脑":           "70",
    "手机":           "70",
    "电讯器材":       "70",

    # Conglomerates
    "企业集团":       "80",
}

# ============================================================================
# SECTION 3: BUILD UNIFIED CHINESE-NAME -> INDUSTRY LOOKUP
# ============================================================================

def _build_name_to_code():
    """Combine all name mappings into a single lookup dict.
    Priority: sub-sector > sector > aliases (order doesn't matter, no conflicts by design).
    """
    lookup = {}
    # Layer 1: Exact sub-sector names (highest specificity)
    lookup.update(HSICS_SUBSECTORS)
    # Layer 2: Sector names
    lookup.update(SECTOR_TO_INDUSTRY)
    # Layer 3: Common aliases
    lookup.update(NAME_ALIASES)
    return lookup

NAME_TO_HSICS = _build_name_to_code()


def get_industry_from_name(name: str) -> str:
    """Map a Chinese industry/sub-sector name to an HSICS industry code string.

    Returns something like '50_金融业', or None if no match.
    """
    code = NAME_TO_HSICS.get(name)
    if code is None:
        return None
    industry_name = HSICS_INDUSTRIES[code][0]
    return f"{code}_{industry_name}"


# ============================================================================
# SECTION 4: VERIFIED AKSHARE DATA
# ============================================================================

def _load_akshare_verified():
    """Load the 10 verified stock-industry mappings from the akshare cache."""
    cache_path = os.path.join(os.path.dirname(__file__), 'hk_industry_map_v2.json')
    verified = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding='utf-8') as f:
            raw = json.load(f)
        for code, ind in raw.items():
            # Filter out errors and unknowns
            if ind == 'Unknown' or ind.startswith('ERR'):
                continue
            # Try to map the Chinese industry name to HSICS code
            hsics = get_industry_from_name(ind)
            if hsics:
                # Normalize code: strip '!1', '!2' suffixes, ensure 5-digit
                clean_code = code.split('!')[0].zfill(5)
                verified[clean_code] = hsics
    return verified

AKSHARE_VERIFIED = _load_akshare_verified()


# ============================================================================
# SECTION 4b: MANUAL OVERRIDES FOR WELL-KNOWN HSCI STOCKS
# ============================================================================
# These are hardcoded based on publicly known HSICS classifications.
# Used when akshare data is unavailable. Overrides the range fallback.

MANUAL_OVERRIDES = {
    # ========================
    # 00 能源业 (Energy)
    # ========================
    "00883": "00_能源业",   # CNOOC 中国海洋石油
    "00857": "00_能源业",   # PetroChina 中国石油股份
    "00386": "00_能源业",   # Sinopec 中国石油化工
    "01088": "00_能源业",   # China Shenhua 中国神华
    "01898": "00_能源业",   # China Coal Energy 中煤能源
    "01171": "00_能源业",   # Yankuang Energy 兖矿能源

    # ========================
    # 05 原材料业 (Materials)
    # ========================
    "02899": "05_原材料业",  # Zijin Mining 紫金矿业
    "02600": "05_原材料业",  # Chalco 中国铝业
    "00358": "05_原材料业",  # Jiangxi Copper 江西铜业
    "01208": "05_原材料业",  # MMG 五矿资源
    "01378": "05_原材料业",  # China Hongqiao 中国宏桥
    "00323": "05_原材料业",  # Maanshan Iron 马鞍山钢铁
    "00914": "05_原材料业",  # Conch Cement 海螺水泥
    "03323": "05_原材料业",  # CNBM 中国建材
    "01313": "05_原材料业",  # CR Building Materials 华润建材科技

    # ========================
    # 10 工业 (Industrials)
    # ========================
    "00669": "10_工业",      # Techtronic Industries 创科实业
    "02382": "10_工业",      # Sunny Optical 舜宇光学科技 (电子零件)
    "02018": "10_工业",      # AAC Tech 瑞声科技 (电子零件)
    "00300": "10_工业",      # Midea Group 美的集团
    "00179": "10_工业",      # Johnson Electric 德昌电机

    # ========================
    # 23 非必需性消费 (Consumer Discretionary)
    # ========================
    "00027": "23_非必需性消费",  # Galaxy Ent 银河娱乐
    "01928": "23_非必需性消费",  # Sands China 金沙中国
    "00880": "23_非必需性消费",  # SJM Holdings 澳博控股
    "01128": "23_非必需性消费",  # Wynn Macau 永利澳门
    "02282": "23_非必需性消费",  # MGM China 美高梅中国
    "00066": "23_非必需性消费",  # MTR 港铁公司 (公共运输)
    "00293": "23_非必需性消费",  # Cathay Pacific 国泰航空

    # ========================
    # 25 必需性消费 (Consumer Staples)
    # ========================
    "00291": "25_必需性消费",  # CR Beer 华润啤酒
    "00168": "25_必需性消费",  # Tsingtao Brewery 青岛啤酒股份
    "01876": "25_必需性消费",  # Budweiser APAC 百威亚太
    "00220": "25_必需性消费",  # Uni-President China 统一企业中国
    "00322": "25_必需性消费",  # Tingyi 康师傅控股
    "01044": "25_必需性消费",  # Hengan International 恒安国际
    "02319": "25_必需性消费",  # Mengniu Dairy 蒙牛乳业
    "01112": "25_必需性消费",  # H&H International 健合集团
    "00151": "25_必需性消费",  # Want Want China 中国旺旺

    # ========================
    # 28 医疗保健业 (Healthcare)
    # ========================
    "02269": "28_医疗保健业",  # WuXi Biologics 药明生物
    "02359": "28_医疗保健业",  # WuXi AppTec 药明康德
    "01177": "28_医疗保健业",  # Sino Biopharm 中国生物制药
    "01093": "28_医疗保健业",  # CSPC Pharma 石药集团
    "01801": "28_医疗保健业",  # Innovent Biologics 信达生物
    "09926": "28_医疗保健业",  # Akeso 康方生物
    "02696": "28_医疗保健业",  # Henlius 复宏汉霖
    "06160": "28_医疗保健业",  # BeiGene 百济神州
    "00013": "28_医疗保健业",  # HUTCHMED 和黄医药
    "00867": "28_医疗保健业",  # CMS 康哲药业

    # ========================
    # 35 电讯业 (Telecommunications)
    # ========================
    "00941": "35_电讯业",      # China Mobile 中国移动
    "00728": "35_电讯业",      # China Telecom 中国电信
    "00762": "35_电讯业",      # China Unicom 中国联通
    "00788": "35_电讯业",      # China Tower 中国铁塔

    # ========================
    # 40 公用事业 (Utilities)
    # ========================
    "00002": "40_公用事业",    # CLP Holdings 中电控股
    "00003": "40_公用事业",    # HK & China Gas 香港中华煤气
    "00006": "40_公用事业",    # Power Assets 电能实业
    "01038": "40_公用事业",    # CKI Holdings 长江基建集团
    "02638": "40_公用事业",    # HK Electric 港灯-SS
    "00836": "40_公用事业",    # CR Power 华润电力
    "00902": "40_公用事业",    # Huaneng Power 华能国际电力
    "02380": "40_公用事业",    # China Power 中国电力

    # ========================
    # 50 金融业 (Financials)
    # ========================
    # Major banks
    "00939": "50_金融业",      # CCB 建设银行
    "01398": "50_金融业",      # ICBC 工商银行
    "03988": "50_金融业",      # Bank of China 中国银行
    "01288": "50_金融业",      # ABC 农业银行
    "03968": "50_金融业",      # CMB 招商银行
    "03328": "50_金融业",      # BOCOM 交通银行
    "01658": "50_金融业",      # PSBC 邮储银行
    "01988": "50_金融业",      # Minsheng Bank 民生银行
    "00998": "50_金融业",      # CITIC Bank 中信银行
    "06818": "50_金融业",      # CEB Bank 光大银行
    "02016": "50_金融业",      # Zheshang Bank 浙商银行
    "02388": "50_金融业",      # BOC Hong Kong 中银香港
    "00011": "50_金融业",      # Hang Seng Bank 恒生银行
    "00023": "50_金融业",      # Bank of East Asia 东亚银行
    "02356": "50_金融业",      # Dah Sing Banking 大新银行集团
    "06198": "50_金融业",      # Qingdao Bank 青岛银行
    "03618": "50_金融业",      # Chongqing Rural Bank 重庆农商行
    # Major insurers
    "02318": "50_金融业",      # Ping An Insurance 中国平安
    "02628": "50_金融业",      # China Life 中国人寿
    "02601": "50_金融业",      # CPIC 中国太保
    "01339": "50_金融业",      # PICC 中国人民保险集团
    "02328": "50_金融业",      # PICC P&C 中国财险
    "01299": "50_金融业",      # AIA 友邦保险
    "06060": "50_金融业",      # ZhongAn Online 众安在线
    # Securities / other financials
    "00388": "50_金融业",      # HKEX 香港交易所
    "06030": "50_金融业",      # CITIC Securities 中信证券
    "06837": "50_金融业",      # Haitong Securities 海通证券
    "01776": "50_金融业",      # GF Securities 广发证券
    "03908": "50_金融业",      # CICC 中金公司
    "06881": "50_金融业",      # CGS 中国银河

    # ========================
    # 60 地产建筑业 (Properties & Construction)
    # ========================
    "00823": "60_地产建筑业",  # Link REIT 领展房产基金
    "01109": "60_地产建筑业",  # China Resources Land 华润置地
    "00688": "60_地产建筑业",  # China Overseas 中国海外发展
    "02007": "60_地产建筑业",  # Country Garden 碧桂园
    "03333": "60_地产建筑业",  # Evergrande 中国恒大
    "00960": "60_地产建筑业",  # Longfor Group 龙湖集团
    "00016": "60_地产建筑业",  # SHK Properties 新鸿基地产
    "00012": "60_地产建筑业",  # Henderson Land 恒基地产
    "00017": "60_地产建筑业",  # New World Dev 新世界发展
    "00101": "60_地产建筑业",  # Hang Lung Properties 恒隆地产
    "00083": "60_地产建筑业",  # Sino Land 信和置业
    "03311": "60_地产建筑业",  # China State Construction 中国建筑国际
    "01800": "60_地产建筑业",  # China Communications 中国交建
    "01186": "60_地产建筑业",  # CRCC 中国铁建
    "00390": "60_地产建筑业",  # CRG 中国中铁
    "03900": "60_地产建筑业",  # Greentown China 绿城中国

    # ========================
    # 70 资讯科技业 (Information Technology)
    # ========================
    "00700": "70_资讯科技业",  # Tencent 腾讯控股
    "09988": "70_资讯科技业",  # Alibaba 阿里巴巴-SW
    "03690": "70_资讯科技业",  # Meituan 美团-W
    "01810": "70_资讯科技业",  # Xiaomi 小米集团-W
    "09618": "70_资讯科技业",  # JD.com 京东集团-SW
    "01024": "70_资讯科技业",  # Kuaishou 快手-W
    "09888": "70_资讯科技业",  # Baidu 百度集团-SW
    "09999": "70_资讯科技业",  # NetEase 网易-S
    "09898": "70_资讯科技业",  # Weibo 微博-SW
    "09961": "70_资讯科技业",  # Trip.com 携程集团-S
    "02015": "70_资讯科技业",  # Li Auto 理想汽车-W
    "09866": "70_资讯科技业",  # NIO 蔚来-SW
    "09868": "70_资讯科技业",  # XPeng 小鹏汽车-W
    "00981": "70_资讯科技业",  # SMIC 中芯国际
    "01347": "70_资讯科技业",  # Hua Hong Semi 华虹半导体
    "00992": "70_资讯科技业",  # Lenovo Group 联想集团

    # ========================
    # 80 综合企业 (Conglomerates)
    # ========================
    "00001": "80_综合企业",    # CK Hutchison 长和
    "00019": "80_综合企业",    # Swire Pacific A 太古股份公司A
    "00087": "80_综合企业",    # Swire Pacific B 太古股份公司B
    "00267": "80_综合企业",    # CITIC Limited 中信股份
}


# ============================================================================
# SECTION 5: STOCK CODE RANGE FALLBACK
# ============================================================================

# Hong Kong stock codes are assigned chronologically, NOT by industry.
# These ranges are approximate fallbacks for stocks without verified/manual data.
# Refined from hsics_final.py based on actual HSCI constituent distributions.

def _code_range_fallback(numeric_code: int) -> str:
    """Fallback industry classification based on HK stock code ranges.

    Covers 00001-09999. Codes >= 10000 are mostly new-economy stocks
    (IT, Healthcare) that were listed in recent years.
    """
    c = numeric_code

    if c < 100:
        # Oldest blue chips: banks, utilities, property, conglomerates
        return "80_综合企业"
    elif 100 <= c < 200:
        # Mix of property, consumer, conglomerate
        return "60_地产建筑业"
    elif 200 <= c < 400:
        # Properties, hotels, consumer goods
        return "60_地产建筑业"
    elif 400 <= c < 500:
        # Financials, various
        return "50_金融业"
    elif 500 <= c < 600:
        # Consumer, retail
        return "23_非必需性消费"
    elif 600 <= c < 700:
        # Industrials, manufacturing
        return "10_工业"
    elif 700 <= c < 800:
        # IT / Telecom starts here (00700 Tencent, 00728 China Telecom)
        return "70_资讯科技业"
    elif 800 <= c < 1000:
        # Properties, consumer, telecom
        return "23_非必需性消费"
    elif 1000 <= c < 1100:
        # Tech / new economy
        return "70_资讯科技业"
    elif 1100 <= c < 1200:
        # Pharmaceuticals, properties, consumer
        return "28_医疗保健业"
    elif 1200 <= c < 1300:
        # Consumer, insurance, construction
        return "23_非必需性消费"
    elif 1300 <= c < 1400:
        # Financials / Insurance
        return "50_金融业"
    elif 1400 <= c < 1500:
        # Industrials, consumer electronics
        return "10_工业"
    elif 1500 <= c < 1600:
        # Healthcare, medical devices
        return "28_医疗保健业"
    elif 1600 <= c < 1800:
        # Industrials, manufacturing, consumer
        return "10_工业"
    elif 1800 <= c < 1900:
        # Construction, infrastructure, new energy
        return "60_地产建筑业"
    elif 1900 <= c < 2000:
        # Financials, consumer, materials
        return "50_金融业"
    elif 2000 <= c < 2100:
        # Consumer, retail, sportswear
        return "23_非必需性消费"
    elif 2100 <= c < 2200:
        # Healthcare services, pharma distribution
        return "28_医疗保健业"
    elif 2200 <= c < 2300:
        # Consumer, food & beverage, pharma
        return "23_非必需性消费"
    elif 2300 <= c < 2400:
        # Consumer, insurance (2318 Ping An misclassified w/o override)
        return "23_非必需性消费"
    elif 2400 <= c < 2500:
        # IT hardware / electronics
        return "70_资讯科技业"
    elif 2500 <= c < 2600:
        # Healthcare (WuXi, etc.) or medical
        return "28_医疗保健业"
    elif 2600 <= c < 2700:
        # Materials, resources, insurance
        return "05_原材料业"
    elif 2700 <= c < 2800:
        # Real estate, properties
        return "60_地产建筑业"
    elif 2800 <= c < 2900:
        # Financials, banks
        return "50_金融业"
    elif 2900 <= c < 3000:
        # Consumer staples
        return "25_必需性消费"
    elif 3000 <= c < 3100:
        # Electronics / IT components
        return "70_资讯科技业"
    elif 3100 <= c < 3300:
        # Industrials, textiles
        return "10_工业"
    elif 3300 <= c < 3400:
        # Materials, properties, financials
        return "05_原材料业"
    elif 3400 <= c < 3700:
        # Industrials, consumer
        return "10_工业"
    elif 3700 <= c < 3800:
        # Consumer, hotels, travel
        return "23_非必需性消费"
    elif 3800 <= c < 3900:
        # Materials, energy equipment
        return "05_原材料业"
    elif 3900 <= c < 4000:
        # Banks (3968 CMB, 3988 BOC), properties (3900 Greentown)
        return "50_金融业"
    elif 4000 <= c < 6000:
        # Manufacturing, industrials (broad range)
        return "10_工业"
    elif 6000 <= c < 6200:
        # Financials, securities
        return "50_金融业"
    elif 6200 <= c < 6300:
        # Consumer, gaming, media
        return "23_非必需性消费"
    elif 6300 <= c < 6400:
        # Industrials
        return "10_工业"
    elif 6400 <= c < 6600:
        # IT / technology hardware
        return "70_资讯科技业"
    elif 6600 <= c < 6700:
        # Consumer staples, healthcare
        return "25_必需性消费"
    elif 6700 <= c < 7000:
        # Properties, construction, healthcare
        return "60_地产建筑业"
    elif 7000 <= c < 8000:
        # New economy tech, internet platforms
        return "70_资讯科技业"
    elif 8000 <= c < 9000:
        return "23_非必需性消费"
    elif 9000 <= c < 10000:
        # New listings, tech, netease, baidu-style codes
        return "70_资讯科技业"
    elif 10000 <= c:
        # Very new listings (2020+), mostly new economy
        return "70_资讯科技业"
    else:
        return "80_综合企业"


# ============================================================================
# SECTION 6: MAIN MAPPING FUNCTION
# ============================================================================

def get_hsics_industry(sid: str) -> str:
    """Map a Hong Kong stock identifier to its HSICS 2-digit industry code.

    Args:
        sid: Stock identifier in formats like:
             - "00005.HK"  (with .HK suffix)
             - "00005"     (bare 5-digit code)
             - "5"         (short numeric code)

    Returns:
        HSICS industry string like "50_金融业", "70_资讯科技业", etc.
        Returns "80_综合企业" if no better match can be found.

    Strategy (priority order):
        1. Verified akshare mapping (10 stocks from cached data)
        2. Manual overrides (~90 well-known HSCI large-caps)
        3. Stock code range fallback (approximate, for remaining stocks)
    """
    # Normalize the code: strip .HK, zero-pad to 5 digits
    clean = sid.replace('.HK', '').replace('.hk', '').strip()
    try:
        code5 = str(int(clean)).zfill(5)
    except (ValueError, TypeError):
        # If we can't parse a numeric code, return conglomerate
        return "80_综合企业"

    # Priority 1: Verified akshare mapping
    if code5 in AKSHARE_VERIFIED:
        return AKSHARE_VERIFIED[code5]

    # Priority 2: Manual overrides for well-known stocks
    if code5 in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[code5]

    # Priority 3: Code range fallback
    try:
        numeric_code = int(code5)
    except ValueError:
        return "80_综合企业"

    return _code_range_fallback(numeric_code)


def get_hsics_code(sid: str) -> str:
    """Return just the 2-digit HSICS industry code (no Chinese name).

    Returns: "00", "05", "10", "23", "25", "28", "35", "40", "50", "60", "70", "80"
    """
    result = get_hsics_industry(sid)
    return result.split('_')[0]


def get_hsics_name(sid: str) -> str:
    """Return just the Chinese industry name (no numeric code).

    Returns: "能源业", "金融业", etc.
    """
    result = get_hsics_industry(sid)
    return result.split('_', 1)[1] if '_' in result else result


# ============================================================================
# SECTION 7: STATISTICS & DIAGNOSTICS (run when executed as script)
# ============================================================================

if __name__ == '__main__':
    from collections import Counter

    print("=" * 80)
    print("HSICS Industry Mapping Statistics")
    print("=" * 80)

    # ---------- Hierarchy overview ----------
    print(f"\n HSICS Hierarchy Overview:")
    print(f"   Total Industries:  12")
    print(f"   Total Sectors:     31")
    print(f"   Total Sub-sectors: 112 (extracted from official PDF)")
    print(f"   Name mappings:     {len(NAME_TO_HSICS)} Chinese names mapped")

    # ---------- Verified akshare data ----------
    print(f"\n Verified akshare records: {len(AKSHARE_VERIFIED)} stocks")
    if AKSHARE_VERIFIED:
        print(f"   Examples:")
        for code, industry in list(AKSHARE_VERIFIED.items())[:12]:
            print(f"     {code}.HK -> {industry}")

    # ---------- Name-to-industry coverage ----------
    print(f"\n Industry code distribution in name mappings:")
    code_counts = Counter(NAME_TO_HSICS.values())
    for code in ["00", "05", "10", "23", "25", "28", "35", "40", "50", "60", "70", "80"]:
        ind_name = HSICS_INDUSTRIES[code][0]
        cnt = code_counts.get(code, 0)
        bar = "#" * cnt
        print(f"   {code}_{ind_name:<20s}: {cnt:>3d} names  {bar}")

    # ---------- Try to load HSCI constituents and show distribution ----------
    hsci_path = os.path.join(os.path.dirname(__file__), 'HSCI.csv')
    if os.path.exists(hsci_path):
        import pandas as pd
        hsci = pd.read_csv(hsci_path)
        sids = hsci['sid'].unique()
        print(f"\n HSCI Constituents: {len(sids)} unique stocks")

        industry_map = {}
        for sid in sids:
            industry_map[sid] = get_hsics_industry(sid)

        dist = Counter(industry_map.values())
        print(f"\n Industry Distribution ({len(industry_map)} stocks):")
        print(f"   {'Industry':<25s} {'Count':>6s}   {'Pct':>6s}")
        print(f"   {'-'*25} {'-'*6}   {'-'*6}")
        for code in ["00", "05", "10", "23", "25", "28", "35", "40", "50", "60", "70", "80"]:
            ind_name = HSICS_INDUSTRIES[code][0]
            key = f"{code}_{ind_name}"
            cnt = dist.get(key, 0)
            pct = cnt / len(industry_map) * 100 if industry_map else 0
            bar = "#" * int(cnt / max(dist.values()) * 40) if dist else ""
            print(f"   {key:<25s} {cnt:>5d}   {pct:>5.1f}%  {bar}")
        print(f"   {'TOTAL':<25s} {len(industry_map):>5d}  {100:>6.1f}%")

        # Show which stocks use which mapping tier
        verified_count = sum(1 for sid in sids
                            if sid.replace('.HK', '').zfill(5) in AKSHARE_VERIFIED)
        manual_count = sum(1 for sid in sids
                          if sid.replace('.HK', '').zfill(5) in MANUAL_OVERRIDES)
        # Only count manual if not already verified
        manual_only_count = sum(1 for sid in sids
                               if sid.replace('.HK', '').zfill(5) not in AKSHARE_VERIFIED
                               and sid.replace('.HK', '').zfill(5) in MANUAL_OVERRIDES)
        fallback_count = len(sids) - verified_count - manual_only_count
        print(f"\n Mapping method:")
        print(f"   Verified akshare:    {verified_count:>4d} stocks (10 known)")
        print(f"   Manual overrides:    {manual_only_count:>4d} stocks (~90 well-known)")
        print(f"   Code-range fallback: {fallback_count:>4d} stocks")
    else:
        print(f"\n [NOTE] HSCI.csv not found at {hsci_path}")
        print(f"   Run from project root to see full HSCI distribution.")

    # ---------- Demonstrate the API ----------
    print(f"\n Usage examples:")
    for sid in ["00005.HK", "00700.HK", "00941.HK", "00011.HK", "02318.HK",
                "03968.HK", "00939.HK", "02269.HK", "00883.HK", "09988.HK"]:
        industry = get_hsics_industry(sid)
        code = get_hsics_code(sid)
        name = get_hsics_name(sid)
        code5 = sid.replace('.HK', '').zfill(5)
        if code5 in AKSHARE_VERIFIED:
            via = "verified"
        elif code5 in MANUAL_OVERRIDES:
            via = "manual"
        else:
            via = "fallback"
        print(f"   {sid:<12s} -> {industry:<22s} (code={code}, name={name}) [{via}]")

    print(f"\n{'=' * 80}")
    print("Done.")
