import asyncio
from glimcrawl import GoogleImageCrawler
from playwright.async_api import async_playwright
#数据工作：https://pypi.org/project/glimcrawl/
#分类依据：https://zh.wikipedia.org/wiki/Category:%E6%97%85%E9%81%8A%E6%99%AF%E9%BB%9E
#现有数据集：https://github.com/koishi70/Landscape-Dataset （风景数据集）


# 关键词列表
attraction_keywords = [
    # 自然景观
    "Citizen Park", "waterfall", "mountain", "beach","river","forest","ocean",
    "lake", "canyon", "valley", "glacier", "volcano", "desert", "island",
    "cave", "cliff", "reef","harbor","city walk", "bridge", "fountain"
]

culture=["castle",  "ancient temple", "monument", "temple", "church",

    # 文化场馆
    "museum", "art gallery", "theater", "concert hall", "opera house",
    "library",  "exhibition hall",  "zoo", "botanical garden"]# 历史人文

# 旅游活动场景
activity_keywords = [
    "hiking", "skiing", "swimming", "sunbathing", "picnic", "camping",
    "shopping", "dining", "photo taking",
]

# 旅游人群
people_keywords = [
    "tourists", "visitors", "crowd", "tour group", "family with kids",
    "couple", "backpacker", "senior traveler", "solo traveler"
]
photo=['selfie','Group photo','Single-person full-body photo']
shopping=["shopping mall","souvenir shop",
    "entertainment district",  "nightclub"]
traffic=["railway", "cable car", "train", "airport"]

# 全部合并
#KEYWORDS = attraction_keywords + activity_keywords + people_keywords+photo+shopping+traffic+culture
KEYWORDS=['shopping', 'reef', 'swimming', 'cliff', 'city walk', 'exhibition hall', 'couple', 'visitors', 'camping', 'art gallery', 'harbor', 'opera house', 'zoo', 'sunbathing', 'castle', 'tourists', 'library', 'lake', 'theater', 'ocean', 'family with kids', 'picnic', 'crowd', 'airport', 'botanical garden', 'canyon', 'bridge', 'glacier', 'backpacker', 'senior traveler', 'cave', 'concert hall', 'dining', 'valley', 'volcano', 'solo traveler', 'train', 'island', 'skiing', 'desert', 'Single-person full-body photo', 'fountain', 'beach', 'selfie', 'hiking', 'photo taking', 'mountain', 'monument', 'railway', 'museum', 'Group photo', 'church', 'temple', 'ancient temple', 'cable car', 'nightclub', 'souvenir shop', 'tour group', 'shopping mall', 'entertainment district', 'waterfall', 'Citizen Park', 'forest']
len(KEYWORDS)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for keyword in KEYWORDS:
            print(f"\n开始下载: {keyword}")
            try:
                # 初始化爬虫时设置参数
                crawler = GoogleImageCrawler(
                    browser=browser,
                    max_images=200,  # 这里设置下载数量
                    save_dir="./images",  # 保存目录
                    use_keyword_dir=True,  # 创建关键词子目录
                    if_exists="rename"  # 重命名如果存在
                )

                # 执行爬取
                result = await crawler.crawl_images(
                    keyword=keyword,
                    size="m",  # 大图
                    date="y"  # 不限制时间
                )

                print(f"✅ {keyword}: 成功 {result.success_count} 张")

            except Exception as e:
                print(f"❌ {keyword} 失败: {e}")


        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
    print("\n🎉 全部完成！")

