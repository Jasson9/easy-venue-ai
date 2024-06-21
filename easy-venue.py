import modal
import pydantic

vol = modal.Volume.from_name("easy-venue")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("pandas", "numpy", "scikit-learn")
)
app = modal.App(name="easy-venue-recommendation-system",volumes={"/easy-venue": vol},image=image)

class Item(pydantic.BaseModel):
    criteria: str


@app.cls(image=image, volumes={"/easy-venue": vol})
class WebApp:
    
    def get_recommendations(self, name, top_n=10):
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        text_vec = self.tfidf.transform([name])
        cosine_sim = cosine_similarity(text_vec, self.tfidf.fit_transform(self.data['categories']))
        idxs = np.argsort(cosine_sim[0])[-top_n:][::-1]
        return self.data.iloc[idxs]['id']

    @modal.enter()
    def startup(self):
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        file_path = '/easy-venue/Bangalore_venues.csv'
        self.data = pd.read_csv(file_path)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf.fit(self.data['categories'])

    @modal.web_endpoint(method="POST")
    def recommend(self,item: Item):
        categories = self.get_recommendations(item.criteria)
        return categories.to_list()