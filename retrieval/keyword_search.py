"""Lightweight BM25-style keyword retrieval."""
import math,re
class KeywordIndex:
    def __init__(self): self.chunks=[]; self.df={}; self.avgdl=0
    def add(self,chunks):
        self.chunks=chunks; self.df={}; total=0
        for c in chunks:
            terms=set(re.findall(r"\w+",c["text"].lower())); total+=len(terms)
            for t in terms:self.df[t]=self.df.get(t,0)+1
        self.avgdl=total/max(1,len(chunks))
    def search(self,query,top_k=5):
        q=set(re.findall(r"\w+",query.lower())); n=len(self.chunks); k1,b=1.5,.75
        scored=[]
        for c in self.chunks:
            terms=re.findall(r"\w+",c["text"].lower()); dl=len(terms)
            score=0
            for t in q:
                tf=terms.count(t)
                if tf: score+=(math.log(1+(n-self.df.get(t,0)+.5)/(self.df.get(t,0)+.5))*tf*(k1+1)/(tf+k1*(1-b+b*dl/max(self.avgdl,1))))
            scored.append((score,c))
        scored.sort(key=lambda x:x[0],reverse=True)
        return [{**c,"score":float(s)} for s,c in scored[:top_k]]
