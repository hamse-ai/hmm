"""Windowed feature extraction and diagonal-Gaussian HMM for Sensor Logger activity data."""
from __future__ import annotations
import csv, math, re
from pathlib import Path
import numpy as np

ACTIVITIES = ["standing", "walking", "jumping", "still"]
FEATURE_NAMES = ["acc_mag_mean", "acc_mag_std", "acc_sma", "gyro_mag_rms", "gyro_mag_std", "acc_dominant_hz", "acc_spectral_energy"]
WINDOW_SECONDS = 2.0
HOP_SECONDS = 1.0
TRAIN_MAX_INDEX = 39


def read_sensor(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    t = np.asarray([float(r["seconds_elapsed"]) for r in rows])
    a = np.asarray([[float(r[axis]) for axis in ("x", "y", "z")] for r in rows])
    if len(t) < 2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(a)):
        raise ValueError(f"Invalid or too-short sensor file: {path}")
    order = np.argsort(t)
    t, a = t[order], a[order]
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], a[keep]


def extract_recording(folder, window_seconds=WINDOW_SECONDS, hop_seconds=HOP_SECONDS):
    ta, acc = read_sensor(folder / "Accelerometer.csv")
    tg, gyro = read_sensor(folder / "Gyroscope.csv")
    start = max(ta[0], tg[0])
    stop = min(ta[-1], tg[-1])
    dt = np.median(np.diff(ta))
    fs = 1.0 / dt
    features, centers = [], []
    left = start
    while left + window_seconds <= stop + 1e-9:
        right = left + window_seconds
        ia = (ta >= left) & (ta < right)
        ig = (tg >= left) & (tg < right)
        if ia.sum() >= max(12, int(fs * window_seconds * .65)) and ig.sum() >= 8:
            av, gv = acc[ia], gyro[ig]
            am = np.linalg.norm(av, axis=1)
            gm = np.linalg.norm(gv, axis=1)
            # Interpolate the acceleration magnitude to a uniform grid before FFT.
            grid = np.arange(left, right, 1.0 / fs)
            sig = np.interp(grid, ta, np.linalg.norm(acc, axis=1))
            sig = sig - sig.mean()
            power = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
            freq = np.fft.rfftfreq(len(sig), d=1.0 / fs)
            band = (freq >= .5) & (freq <= min(12.0, fs / 2))
            dominant = float(freq[band][np.argmax(power[band])]) if band.any() else 0.0
            energy = float(power[band].sum() / max(1, len(sig))) if band.any() else 0.0
            features.append([am.mean(), am.std(), np.abs(av).sum(axis=1).mean(), np.sqrt(np.mean(gm ** 2)), gm.std(), dominant, energy])
            centers.append((left + right) / 2)
        left += hop_seconds
    if not features:
        raise ValueError(f"No complete windows in {folder}")
    return np.asarray(features, dtype=float), np.asarray(centers), fs, stop - start


def load_dataset(data_root):
    records = []
    for label in ACTIVITIES:
        for folder in sorted((Path(data_root) / label).iterdir()):
            if not folder.is_dir() or not (folder / "Accelerometer.csv").exists():
                continue
            m = re.match(r"[a-z]+(\d+)-", folder.name)
            if not m:
                continue
            idx = int(m.group(1))
            X, times, fs, duration = extract_recording(folder)
            records.append({"activity": label, "index": idx, "recording": folder.name, "folder": folder, "X": X, "times": times, "fs": fs, "duration": duration})
    return records


def logsumexp(a, axis=None):
    mx = np.max(a, axis=axis, keepdims=True)
    out = mx + np.log(np.exp(a - mx).sum(axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    else:
        out = np.squeeze(out)
    return out


class DiagonalGaussianHMM:
    """Four-state diagonal Gaussian HMM with log-space Baum-Welch and Viterbi."""
    def __init__(self, n_states=4, max_iter=100, tol=1e-4, variance_floor=1e-3, transition_prior=0.05):
        self.n_states=n_states; self.max_iter=max_iter; self.tol=tol
        self.variance_floor=variance_floor; self.transition_prior=transition_prior

    def _log_emissions(self, X):
        d=X[:,None,:]-self.means[None,:,:]
        return -.5*(self.n_features*np.log(2*np.pi)+np.log(self.vars).sum(axis=1)[None,:]+(d*d/self.vars[None,:,:]).sum(axis=2))

    def _forward_backward(self, X):
        le=self._log_emissions(X); la=np.log(self.A); lp=np.log(self.pi)
        T=len(X); alpha=np.empty((T,self.n_states)); alpha[0]=lp+le[0]
        for t in range(1,T): alpha[t]=le[t]+logsumexp(alpha[t-1][:,None]+la,axis=0)
        ll=float(logsumexp(alpha[-1]))
        beta=np.zeros((T,self.n_states))
        for t in range(T-2,-1,-1): beta[t]=logsumexp(la+le[t+1][None,:]+beta[t+1][None,:],axis=1)
        log_gamma=alpha+beta-ll
        gamma=np.exp(log_gamma)
        xi_sum=np.zeros((self.n_states,self.n_states))
        for t in range(T-1):
            z=alpha[t][:,None]+la+le[t+1][None,:]+beta[t+1][None,:]-ll
            xi_sum += np.exp(z)
        return ll,gamma,xi_sum

    def fit(self, sequences, labels):
        allx=np.concatenate(sequences); self.n_features=allx.shape[1]
        labels=np.asarray(labels)
        self.means=np.vstack([np.concatenate([x for x,y in zip(sequences,labels) if y==s]).mean(axis=0) for s in range(self.n_states)])
        self.vars=np.vstack([np.concatenate([x for x,y in zip(sequences,labels) if y==s]).var(axis=0) for s in range(self.n_states)])
        self.vars=np.maximum(self.vars, self.variance_floor)
        self.pi=np.ones(self.n_states)/self.n_states
        self.A=np.full((self.n_states,self.n_states), self.transition_prior)
        np.fill_diagonal(self.A, 1.0)
        self.A /= self.A.sum(axis=1,keepdims=True)
        self.history=[]
        previous=-np.inf
        for _ in range(self.max_iter):
            gammas=[]; xis=[]; ll_total=0.0
            for X in sequences:
                ll,g,xi=self._forward_backward(X); ll_total+=ll; gammas.append(g); xis.append(xi)
            self.history.append(ll_total)
            init=np.ones(self.n_states)*self.transition_prior
            trans=np.ones((self.n_states,self.n_states))*self.transition_prior
            weights=np.zeros(self.n_states); sums=np.zeros_like(self.means); sums2=np.zeros_like(self.means)
            for X,g,xi in zip(sequences,gammas,xis):
                init += g[0]; trans += xi
                weights += g.sum(axis=0); sums += g.T@X; sums2 += g.T@(X*X)
            self.pi=init/init.sum(); self.A=trans/trans.sum(axis=1,keepdims=True)
            self.means=sums/np.maximum(weights[:,None],1e-12)
            self.vars=np.maximum(sums2/np.maximum(weights[:,None],1e-12)-self.means**2,self.variance_floor)
            if np.isfinite(previous) and abs(ll_total-previous) <= self.tol*(1+abs(previous)):
                break
            previous=ll_total
        return self

    def viterbi(self, X):
        le=self._log_emissions(X); la=np.log(self.A); lp=np.log(self.pi)
        score=np.empty((len(X),self.n_states)); back=np.zeros((len(X),self.n_states),dtype=int)
        score[0]=lp+le[0]
        for t in range(1,len(X)):
            candidates=score[t-1][:,None]+la
            back[t]=np.argmax(candidates,axis=0)
            score[t]=le[t]+np.max(candidates,axis=0)
        path=np.empty(len(X),dtype=int); path[-1]=int(np.argmax(score[-1]))
        for t in range(len(X)-2,-1,-1): path[t]=back[t+1,path[t+1]]
        return path


def zscore(train, other):
    mean=np.concatenate(train).mean(axis=0); sd=np.concatenate(train).std(axis=0)
    sd[sd < 1e-9]=1.0
    return [(x-mean)/sd for x in train], [(x-mean)/sd for x in other], mean, sd


def evaluate(records, out_dir):
    train=[r for r in records if r["index"] < TRAIN_MAX_INDEX+1]
    test=[r for r in records if r["index"] >= TRAIN_MAX_INDEX+1]
    Xtrain, Xtest, mean, sd=zscore([r["X"] for r in train],[r["X"] for r in test])
    state_ids={a:i for i,a in enumerate(ACTIVITIES)}
    model=DiagonalGaussianHMM().fit(Xtrain,[state_ids[r["activity"]] for r in train])
    y_true=[]; y_pred=[]; per_record=[]
    for r,X in zip(test,Xtest):
        pred=model.viterbi(X); truth=state_ids[r["activity"]]
        y_true.extend([truth]*len(pred)); y_pred.extend(pred.tolist())
        per_record.append({"recording":r["recording"],"activity":r["activity"],"windows":len(pred),"correct":int((pred==truth).sum()),"accuracy":float((pred==truth).mean()),"predicted":pred.tolist()})
    n=len(ACTIVITIES); cm=np.zeros((n,n),dtype=int)
    for t,p in zip(y_true,y_pred): cm[t,p]+=1
    total=cm.sum(); accuracy=float(np.trace(cm)/total)
    metrics=[]
    for i,a in enumerate(ACTIVITIES):
        tp=cm[i,i]; fn=cm[i,:].sum()-tp; fp=cm[:,i].sum()-tp; tn=total-tp-fn-fp
        metrics.append({"activity":a,"n_test_windows":int(cm[i,:].sum()),"sensitivity":tp/(tp+fn) if tp+fn else 0.,"specificity":tn/(tn+fp) if tn+fp else 0.,"accuracy_ovr":(tp+tn)/total if total else 0.,"precision":tp/(tp+fp) if tp+fp else 0.})
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    with (out/"metrics.csv").open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=metrics[0].keys()); w.writeheader(); w.writerows(metrics)
    with (out/"confusion_matrix.csv").open('w',newline='') as f:
        w=csv.writer(f); w.writerow(["true\\pred"]+ACTIVITIES)
        for a,row in zip(ACTIVITIES,cm): w.writerow([a]+row.tolist())
    with (out/"test_recordings.csv").open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=per_record[0].keys()); w.writeheader(); w.writerows(per_record)
    save_heatmap(out/"transition_probabilities.svg",model.A,ACTIVITIES,"HMM transition probabilities",ACTIVITIES)
    sample=next(r for r in records if r["activity"]=="walking")
    raw_t,raw_a=read_sensor(sample["folder"] / "Accelerometer.csv")
    save_raw_plot(out/"raw_accelerometer_sample.svg",raw_t-raw_t[0],raw_a)
    save_heatmap(out/"emission_variances.svg",model.vars,ACTIVITIES,"Learned diagonal emission variances",FEATURE_NAMES)
    save_emission_curves(out/"emission_probabilities.svg",model,ACTIVITIES,FEATURE_NAMES)
    save_heatmap(out/"confusion_matrix.svg",cm,ACTIVITIES,"Unseen test confusion matrix",ACTIVITIES,integer=True)
    save_sequence_plot(out/"decoded_sequences.svg",per_record)
    return model, cm, metrics, accuracy, len(train),len(test)


def save_heatmap(path, matrix, rows, title, cols, integer=False):
    m=np.asarray(matrix,dtype=float); width=920; left=210; top=90; cw=min(95,650/len(cols)); ch=62
    vals=m.copy(); lo,hi=float(vals.min()),float(vals.max()); span=hi-lo if hi>lo else 1.0
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{top+ch*len(rows)+80}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{width/2}" y="35" text-anchor="middle" font-family="Arial" font-size="22">{title}</text>']
    for j,c in enumerate(cols): s.append(f'<text x="{left+j*cw+cw/2}" y="{top-10}" text-anchor="middle" font-family="Arial" font-size="12">{c}</text>')
    for i,r in enumerate(rows):
        s.append(f'<text x="{left-12}" y="{top+i*ch+ch/2+4}" text-anchor="end" font-family="Arial" font-size="13">{r}</text>')
        for j in range(m.shape[1]):
            v=m[i,j]; q=(v-lo)/span; red=int(247-160*q); green=int(250-130*q); blue=int(252-35*q)
            x=left+j*cw; y=top+i*ch
            lab=f'{v:.2f}' if not integer else str(int(v))
            s.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="rgb({red},{green},{blue})" stroke="white"/>')
            s.append(f'<text x="{x+cw/2}" y="{y+ch/2+5}" text-anchor="middle" font-family="Arial" font-size="13">{lab}</text>')
    s.append('</svg>'); Path(path).write_text('\n'.join(s),encoding='utf8')


def save_raw_plot(path, times, values):
    keep=times <= 5.0
    t=times[keep]; v=values[keep]
    width=950; height=360; left=75; top=55; pw=820; ph=240
    lo=float(v.min()); hi=float(v.max()); pad=(hi-lo)*.08 or 1.0; lo-=pad; hi+=pad
    colors=["#4e79a7","#e15759","#59a14f"]; names=["x","y","z"]
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">','<rect width="100%" height="100%" fill="white"/>','<text x="475" y="28" text-anchor="middle" font-family="Arial" font-size="20">Raw accelerometer axes: sample walking recording</text>',f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#555"/>',f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#555"/>']
    for i,n in enumerate(names):
        pts=[]
        for ti,vi in zip(t,v[:,i]):
            x=left+ti/5*pw; y=top+ph-(vi-lo)/(hi-lo)*ph; pts.append(f'{x:.1f},{y:.1f}')
        s.append(f'<polyline fill="none" stroke="{colors[i]}" stroke-width="1.4" points="{" ".join(pts)}"/>')
        s.append(f'<line x1="{left+pw-140}" y1="{top+15*i}" x2="{left+pw-120}" y2="{top+15*i}" stroke="{colors[i]}" stroke-width="3"/><text x="{left+pw-115}" y="{top+15*i+4}" font-family="Arial" font-size="12">{n}</text>')
    s.append(f'<text x="{left+pw/2}" y="{top+ph+32}" text-anchor="middle" font-family="Arial" font-size="13">Elapsed time (seconds)</text>')
    s.append('</svg>'); Path(path).write_text('\n'.join(s),encoding='utf8')


def save_emission_curves(path, model, labels, names):
    # Gaussian density curves for three interpretable standardized features.
    selected=[1,3,5]; colors=["#4e79a7","#59a14f","#e15759","#b07aa1"]
    width=980; panel_h=230; height=panel_h*len(selected)+55
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">','<rect width="100%" height="100%" fill="white"/>','<text x="490" y="28" text-anchor="middle" font-family="Arial" font-size="21">Gaussian emission probability densities (standardized features)</text>']
    for pi,fi in enumerate(selected):
        y0=50+pi*panel_h; left=85; top=y0+25; pw=690; ph=145
        all_lo=min(model.means[:,fi]-3*np.sqrt(model.vars[:,fi])); all_hi=max(model.means[:,fi]+3*np.sqrt(model.vars[:,fi]))
        xs=np.linspace(all_lo,all_hi,180); maxpdf=max(np.max(np.exp(-.5*(xs-mu)**2/var)/np.sqrt(2*np.pi*var)) for mu,var in zip(model.means[:,fi],model.vars[:,fi]))
        s.append(f'<text x="{left}" y="{y0+15}" font-family="Arial" font-size="15">{names[fi]}</text>')
        s.append(f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#555"/>')
        for st,label in enumerate(labels):
            mu=model.means[st,fi]; var=model.vars[st,fi]
            pdf=np.exp(-.5*(xs-mu)**2/var)/np.sqrt(2*np.pi*var)
            points=[]
            for x,y in zip(xs,pdf): points.append(f'{left+(x-all_lo)/(all_hi-all_lo)*pw:.1f},{top+ph-y/maxpdf*(ph-8):.1f}')
            s.append(f'<polyline fill="none" stroke="{colors[st]}" stroke-width="2" points="{" ".join(points)}"/>')
        for st,label in enumerate(labels):
            s.append(f'<line x1="{left+pw+18}" y1="{top+20*st}" x2="{left+pw+38}" y2="{top+20*st}" stroke="{colors[st]}" stroke-width="3"/>')
            s.append(f'<text x="{left+pw+44}" y="{top+20*st+4}" font-family="Arial" font-size="12">{label}</text>')
        s.append(f'<text x="{left+pw/2}" y="{top+ph+20}" text-anchor="middle" font-family="Arial" font-size="11">z-score</text>')
    s.append('</svg>'); Path(path).write_text('\n'.join(s),encoding='utf8')


def save_sequence_plot(path, rows):
    width=1060; left=350; tile=48; gap=2; row_h=25; height=80+len(rows)*row_h
    colors={"standing":"#4e79a7","walking":"#59a14f","jumping":"#e15759","still":"#b07aa1"}
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">','<rect width="100%" height="100%" fill="white"/>','<text x="530" y="28" text-anchor="middle" font-family="Arial" font-size="20">Viterbi-decoded states across held-out recording windows</text>']
    for ri,r in enumerate(rows):
        y=55+ri*row_h; s.append(f'<text x="{left-10}" y="{y+15}" text-anchor="end" font-family="Arial" font-size="10">{r["recording"][:34]}</text>')
        s.append(f'<text x="{left+len(r["predicted"])*(tile+gap)+6}" y="{y+14}" font-family="Arial" font-size="10">true: {r["activity"]}</text>')
        for wi,pred in enumerate(r['predicted']):
            x=left+wi*(tile+gap); color=colors[ACTIVITIES[pred]]
            s.append(f'<rect x="{x}" y="{y}" width="{tile}" height="18" fill="{color}"/>')
            s.append(f'<text x="{x+tile/2}" y="{y+13}" text-anchor="middle" fill="white" font-family="Arial" font-size="9">{ACTIVITIES[pred]}</text>')
    s.append('</svg>'); Path(path).write_text('\n'.join(s),encoding='utf8')


def main(data_root="data", out_dir="results"):
    records=load_dataset(data_root)
    print("Recordings per activity:", {a:sum(r['activity']==a for r in records) for a in ACTIVITIES})
    print("Windows: 2.0 s with 1.0 s hop; splits by recording index (0-39 train, 40-49 unseen test).")
    model,cm,metrics,acc,ntrain,ntest=evaluate(records,out_dir)
    print(f"Train recordings: {ntrain}; unseen test recordings: {ntest}; test windows: {cm.sum()}; accuracy: {acc:.4f}")
    print("Baum-Welch iterations:",len(model.history),"final training log likelihood:",round(model.history[-1],3))
    for row in metrics: print(row)
    print("Confusion matrix (rows=true, columns=predicted):\n",cm)
    print("Wrote results and SVG figures to",out_dir)

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "data", sys.argv[2] if len(sys.argv)>2 else "results")
