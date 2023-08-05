import sqlite3
from recipe_scrapers import scrape_me

def scrape_and_store_recipe(recipe_url):
    print(f"Scraping recipe from: {recipe_url}")
    scraper = scrape_me(recipe_url)
    title = scraper.title()
    ingredients = '\n'.join(scraper.ingredients())

    conn = sqlite3.connect('recipes.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY,
            title TEXT,
            ingredients TEXT
        )
    ''')

    c.execute('INSERT INTO recipes (title, ingredients) VALUES (?, ?)', (title, ingredients))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    recipe_urls = [
        "https://www.allrecipes.com/recipe/72657/best-hamburger-ever/",
        "https://www.allrecipes.com/recipe/8543770/smash-burgers/",
        "https://www.allrecipes.com/recipe/8513919/cheeseburger-wellington/",
        "https://www.allrecipes.com/recipe/49404/juiciest-hamburgers-ever/",
        "https://www.allrecipes.com/recipe/20040/seasoned-turkey-burgers/",
        "https://www.allrecipes.com/recipe/39748/actually-delicious-turkey-burgers/",
        "https://www.allrecipes.com/recipe/25473/the-perfect-basic-burger/",
        "https://www.allrecipes.com/recipe/14497/portobello-mushroom-burgers/",
        "https://www.allrecipes.com/recipe/85452/homemade-black-bean-veggie-burgers/",
        "https://www.allrecipes.com/recipe/272767/chef-johns-loco-moco/"
    ]

    for recipe_url in recipe_urls:
        scrape_and_store_recipe(recipe_url)