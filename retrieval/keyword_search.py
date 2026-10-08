"""Dependency-free BM25 lexical retrieval."""
import math,re
_TOKEN_RE=re.compile(r"(?u)\b\w+\b")
class KeywordIndex:
    def __init__(self): self.chunks=[]; self.df={}; self.avgdl=0.; self._terms=[]
    def add(self,chunks):
        self.chunks=list(chunks); self.df={}; self._terms=[]; total=0
        for c in self.chunks:
            terms=_TOKEN_RE.findall(c["text"].lower()); self._terms.append(terms); total+=len(terms)
            for t in set(terms): self.df[t]=self.df.get(t,0)+1
        self.avgdl=total/max(1,len(self.chunks))
    def search(self,query,top_k=5):
        if top_k<=0:return []
        q=_TOKEN_RE.findall(query.lower()); n=len(self.chunks); k1,b=1.5,.75; scored=[]
        for c,terms in zip(self.chunks,self._terms):
            dl=len(terms); score=0.
            for t in q:
                tf=terms.count(t); df=self.df.get(t,0)
                if tf and df:
                    idf=math.log(1+(n-df+.5)/(df+.5)); score+=idf*tf*(k1+1)/(tf+k1*(1-b+b*dl/max(self.avgdl,1)))
            scored.append((score,c))
        scored.sort(key=lambda x:x[0],reverse=True)
        return [{**c,"score":float(s)} for s,c in scored[:top_k] if s>0]