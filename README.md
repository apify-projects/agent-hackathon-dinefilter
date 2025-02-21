## What does DinoFilter do?
DinoFilter is an Apify actor that scrapes Google Maps search results based on a user prompt, which may include various preferences and location. It then analyzes restaurant reviews, comparing the best and worst feedback, and summarizes key insights from the reviews. Additionally, DinoFilter scrapes restaurant menu websites for additional details and recommends the top three restaurants based on the reviews and user's criteria.

DinoFilter can scrape:
- Google Maps search results
- Restaurant reviews, cuisine type, price etc.
- Restaurant menu details
- Top best and worst restaurant recommendations

## Why scrape Google Maps?
Google Maps is a widely used platform with a vast amount of restaurant information and reviews. By scraping Google Maps, users can gather valuable insights into the quality of restaurants and make informed dining decisions.

Here are just some of the ways you could use the data from DinoFilter:
- Finding the best restaurants in a specific location
- Comparing reviews to make an informed choice
- Analyzing menu offerings for different restaurants
- Personalizing dining experiences based on preferences

If you would like more inspiration on how scraping Google Maps could benefit your business or organization, check out our [industry pages](https://apify.com/industries).

## How to scrape Google Maps with DinoFilter
It's easy to scrape Google Maps search results with DinoFilter. Just follow these simple steps and you'll have access to valuable restaurant data in no time.

1. Click on Try for free.
2. Enter your search criteria, including preferences and location.
3. Click on Run.
4. Once DinoFilter has finished, preview or download your data from the Dataset tab.

## How much will it cost to scrape Google Maps with DinoFilter?
Apify provides $5 free usage credits every month on the [Apify Free plan](https://apify.com/pricing). With DinoFilter, you can get more then 30 results per month for free!

If you need more data regularly from Google Maps, consider upgrading to an Apify subscription for example our [$49/month Starter plan](https://apify.com/pricing) or [Scale plan](https://apify.com/pricing).

## Results
An example of the data produced by DinoFilter:
```json
{
  "name": "Mopa Döner",
  "address": "3 Rue Pierre Fontaine, 75009 Paris, France",
  "location": "48.8826, 2.3370",
  "rating": "5",
  "price_level": "N/A",
  "cuisine": [
    "Vegan",
    "Doner Kebab"
  ],
  "map_url": "https://www.google.com/maps/search/?api=1&query=Mopa%20D%C3%B6ner&query_place_id=ChIJBf3w9hZv5kcRoQWPpk5Xb5w",
  "menu_url": "https://www.instagram.com/p/Cu4lBQtosnI/?img_index=1",
  "web_url": "http://www.mopa-concept.fr/",
  "review_summary": "The restaurant is popular for its vegan offerings. Customers appreciate the taste and texture of the food, particularly the vegan doner and the 'fake chicken'. The staff is welcoming and friendly.",
  "menu_highlights": "Vegan Doner, Fake Chicken"
}
```

## Tips for scraping Google Maps
- Make sure to provide clear and specific search criteria to get the most relevant results.
- Regularly update your search preferences to discover new and trending restaurants in your area.

## Is it legal to scrape Google Maps?
Note that scraping personal data is subject to regulations such as GDPR in the European Union. Ensure that you have a legitimate reason for scraping data and comply with relevant laws and regulations. If in doubt, consult legal counsel to ensure compliance.

For more information on the legality of web scraping, read our blog post: [Is Web Scraping Legal?](https://blog.apify.com/is-web-scraping-legal/).