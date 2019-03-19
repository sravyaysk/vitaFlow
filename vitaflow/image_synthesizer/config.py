import os

merchant_fonts = []
line_items_fonts = []


OUT_DIR = os.environ['DEMO_DATA_PATH'] + "/receipt_mock_data/"

ALL_MERCHANTS_NAMES = '''
Publix
Wegmans
Trader Joe's
H-E-B
Alladin
Harris Teeter
Hy-Vee
Costco
WinCo 
Whole Foods
Bil Fry's
Kroger
Target
Winn-Dixie
ShopRite
Food Lion
Albertsons
Meijer Er
Sam's Club
Giant Food
Safeway
Stop & Shop
Walmart
'''.strip().splitlines()

ALL_LINE_ITEMS = '''
All-purpose flour
American cheese
Apples 
Banana 
Beef Round
Boneless chicken breast
Boneless pork chop
Broccoli
Chicken Breasts
Chocolate chip cookies
Creamy peanut butter
Dried beans
Eggs (regular) 
Frozen turkey
Ground beef
Ice cream
Lemons
Lettuce 
Loaf of Fresh White Bread 
Local Cheese
Milk (regular)
Navel oranges
Onion 
Oranges 
Pasta
Potato 
Rice (white)
Salted butter
Seedless grapes
Sirloin steak
Sliced bacon
Strawberries
Sugar
Tomato 
Top round steak
Wheat bread
Amla
Avarekai
Baby Corn
Baby Corn Cleaned
Basale Greens
Beans
Beet Root
Bitter Gourd
Bottle Gourd
Brinjal [R]
Brinjal [W]
Brinjal Bottle
Cabbage
Cabbage Red
Capsicum
Capsicum Red/Yellow
Carrot Delhi
Carrot Nati
Carrot Ooty
Cauliflower Per Kg
Cauliflower[B]
Chakota Greens
Chillies Bajji
Chillies Green
Chow-Chow
Cluster Bean Local
Coconut [B]
Coconut [M]
Coconut [S]
Corriander Leave
Cucumber
Curry Leave
Dhantu Greens
Double Beans
Drumstik
Eggs
Garlic
Ginger New
Greens Sabbakki
Ground Nut Local
Herali Kai
Knol-Khol
Ladys Finger
Lime Local
Little Gourd
Mangalore Cucumber
Mango Raw
Menthya Greens
Mint Leaves
Mushroom Button
Mushroom Oyster
Onion Medium
Palak Greens
Peas Delhi
Peas Local
Peas Seeds
Potato[M]
Pumpkin Ash
Pumpkin Japan
Raddish
Raddish Red
Sham Gadde
Snake Gourd
Snake Gourd[S]
Sponge Gourd
Spring Onion
Spring Onion[Cleaned]
Sweet Corn
Sweet Corn Seeds
Tender Coconut
Tomoto
Yam/S.Root
Anjura/Fig
'''.strip().splitlines()

ALL_MERCHANTS_ADDR = '''Sanger, CA
Reedley, CA
Temperance & Hwy 168 Clovis, CA
Ashlan & Fowler Clovis, CA
Herndon & Fowler Clovis, CA
Vons Clovis 1756, CA
Sierra Vista Mall Clovis, CA
Vons Fresno 1754, CA
Clovis & Kings Canyon Fresno, CA
Dinuba, CA
Herndon & Clovis Clovis, CA
West Shaw Clovis, CA
Peach & Mckinley Fresno, CA
Target Clovis T 2018, CA
Herndon & Willow Clovis, CA
Willow & Nees Clovis, CA
Frank Phillips & US 75, Bartlesville
Food Pyramid Bartlesville O, OK
Apache Street, Tulsa
Harry & Rock, Wichita
3rd & Range Line, Joplin
Price Cutter Joplin 57, MO
Food Pyramid Tulsa 63, OK
Super Target Tulsa St 1782, OK
51st & Harvard, Tulsa
Aspen & Kenosha, Broken Arrow
Food Pyramid Tulsa 61, OK
Food Pyramid Tulsa 64, OK
Target Broken Arrow T 2422, OK
'''.strip().splitlines()


ALL_LINE_ITEMS = [_.strip() for _ in ALL_LINE_ITEMS]
ALL_MERCHANTS_NAMES = [_.strip() for _ in ALL_MERCHANTS_NAMES]
ALL_MERCHANTS_ADDR = [_.strip() for _ in ALL_MERCHANTS_ADDR]
