"""Windowed feature extraction and diagonal-Gaussian HMM for Sensor Logger activity data."""
from __future__ import annotations
import csv, json, re
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
    """Numerically stable log(sum(exp(a))) with defined all-infinite behavior."""
    a = np.asarray(a, dtype=float)
    mx = np.max(a, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        shifted = np.where(np.isfinite(mx), a - mx, -np.inf)
        total = np.exp(shifted).sum(axis=axis, keepdims=True)
        out = mx + np.log(total)
    out = np.where(np.isneginf(mx), -np.inf, out)
    out = np.where(np.isposinf(mx), np.inf, out)
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    else:
        out = np.squeeze(out)
    return out


class DiagonalGaussianHMM:
    """Diagonal-Gaussian HMM trained with Baum-Welch and decoded by Viterbi.

    The activity labels are used to initialize one Gaussian emission per state.
    Baum-Welch then refines the initial-state, transition, and emission parameters
    across the training sequences. Each sequence remains an independent recording.
    """

    def __init__(
        self,
        n_states=4,
        max_iter=100,
        tol=1e-4,
        variance_floor=1e-3,
        transition_prior=0.05,
    ):
        if n_states < 1 or max_iter < 1 or tol <= 0 or variance_floor <= 0 or transition_prior <= 0:
            raise ValueError("HMM dimensions, iteration limit, tolerance, variance floor, and prior must be positive")
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.variance_floor = variance_floor
        self.transition_prior = transition_prior

    def _log_emissions(self, X):
        """Return log p(x_t | state) for independent per-feature Gaussians."""
        difference = X[:, None, :] - self.means[None, :, :]
        log_normalizer = self.n_features * np.log(2 * np.pi) + np.log(self.vars).sum(axis=1)
        quadratic = (difference * difference / self.vars[None, :, :]).sum(axis=2)
        return -0.5 * (log_normalizer[None, :] + quadratic)

    def _forward_backward(self, X):
        """Compute sequence log-likelihood, state posteriors, and transition counts."""
        log_emission = self._log_emissions(X)
        log_transition = np.log(self.A)
        log_initial = np.log(self.pi)
        time_steps = len(X)

        # Forward recursion: alpha[t, j] sums paths ending in state j.
        alpha = np.empty((time_steps, self.n_states))
        alpha[0] = log_initial + log_emission[0]
        for t in range(1, time_steps):
            alpha[t] = log_emission[t] + logsumexp(
                alpha[t - 1][:, None] + log_transition, axis=0
            )
        log_likelihood = float(logsumexp(alpha[-1]))

        # Backward recursion: beta[t, i] sums all future observations after state i.
        beta = np.zeros((time_steps, self.n_states))
        for t in range(time_steps - 2, -1, -1):
            beta[t] = logsumexp(
                log_transition
                + log_emission[t + 1][None, :]
                + beta[t + 1][None, :],
                axis=1,
            )

        # gamma gives state occupancy; xi gives expected adjacent-state counts.
        gamma = np.exp(alpha + beta - log_likelihood)
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        xi_sum = np.zeros((self.n_states, self.n_states))
        for t in range(time_steps - 1):
            log_xi = (
                alpha[t][:, None]
                + log_transition
                + log_emission[t + 1][None, :]
                + beta[t + 1][None, :]
                - log_likelihood
            )
            xi_sum += np.exp(log_xi)
        return log_likelihood, gamma, xi_sum

    def fit(self, sequences, labels):
        """Initialize from labeled sequences, then refine parameters with EM."""
        if not sequences or len(sequences) != len(labels):
            raise ValueError("Provide at least one sequence and exactly one label per sequence")
        sequences = [np.asarray(sequence, dtype=float) for sequence in sequences]
        labels = np.asarray(labels, dtype=int)
        if np.any((labels < 0) | (labels >= self.n_states)):
            raise ValueError("Sequence labels must be state indices from 0 to n_states - 1")
        if set(labels.tolist()) != set(range(self.n_states)):
            raise ValueError("Training data must contain at least one sequence for every state")
        if any(x.ndim != 2 or len(x) == 0 for x in sequences):
            raise ValueError("Each observation sequence must be a nonempty 2D array")
        n_features = sequences[0].shape[1]
        if any(x.shape[1] != n_features or not np.all(np.isfinite(x)) for x in sequences):
            raise ValueError("Sequences must have matching feature counts and finite values")

        self.n_features = n_features
        # Supervised initialization anchors state identities to the activity labels.
        self.means = np.vstack([
            np.concatenate([x for x, label in zip(sequences, labels) if label == state]).mean(axis=0)
            for state in range(self.n_states)
        ])
        self.vars = np.vstack([
            np.concatenate([x for x, label in zip(sequences, labels) if label == state]).var(axis=0)
            for state in range(self.n_states)
        ])
        self.vars = np.maximum(self.vars, self.variance_floor)

        # Start equally likely; initialize transitions with a self-transition bias.
        self.pi = np.full(self.n_states, 1.0 / self.n_states)
        self.A = np.full((self.n_states, self.n_states), self.transition_prior)
        np.fill_diagonal(self.A, 1.0)
        self.A /= self.A.sum(axis=1, keepdims=True)

        self.history = []
        self.converged = False
        self.stop_reason = "maximum iterations reached"
        previous_log_likelihood = None
        for _ in range(self.max_iter):
            # E step: posterior state occupancy and expected transition counts.
            gammas, xis = [], []
            total_log_likelihood = 0.0
            for sequence in sequences:
                ll, gamma, xi = self._forward_backward(sequence)
                total_log_likelihood += ll
                gammas.append(gamma)
                xis.append(xi)
            self.history.append(total_log_likelihood)

            # M step: update pi, A, and diagonal Gaussian emission parameters.
            initial_counts = np.full(self.n_states, self.transition_prior)
            transition_counts = np.full(
                (self.n_states, self.n_states), self.transition_prior
            )
            state_weights = np.zeros(self.n_states)
            weighted_sum = np.zeros_like(self.means)
            weighted_square_sum = np.zeros_like(self.means)
            for sequence, gamma, xi in zip(sequences, gammas, xis):
                initial_counts += gamma[0]
                transition_counts += xi
                state_weights += gamma.sum(axis=0)
                weighted_sum += gamma.T @ sequence
                weighted_square_sum += gamma.T @ (sequence * sequence)

            self.pi = initial_counts / initial_counts.sum()
            self.A = transition_counts / transition_counts.sum(axis=1, keepdims=True)
            self.means = weighted_sum / np.maximum(state_weights[:, None], 1e-12)
            second_moment = weighted_square_sum / np.maximum(state_weights[:, None], 1e-12)
            self.vars = np.maximum(second_moment - self.means * self.means, self.variance_floor)

            if previous_log_likelihood is not None:
                change = abs(total_log_likelihood - previous_log_likelihood)
                if change <= self.tol * (1 + abs(previous_log_likelihood)):
                    self.converged = True
                    self.stop_reason = "relative log-likelihood tolerance reached"
                    break
            previous_log_likelihood = total_log_likelihood

        self.iterations = len(self.history)
        self.final_log_likelihood = float(
            sum(self._forward_backward(sequence)[0] for sequence in sequences)
        )
        return self

    def viterbi(self, X):
        """Return the most likely hidden state path for one observation sequence."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0 or X.shape[1] != self.n_features or not np.all(np.isfinite(X)):
            raise ValueError("Viterbi input must be a nonempty finite 2D feature sequence")
        log_emission = self._log_emissions(X)
        log_transition = np.log(self.A)
        log_initial = np.log(self.pi)
        time_steps = len(X)
        best_score = np.empty((time_steps, self.n_states))
        backpointer = np.zeros((time_steps, self.n_states), dtype=int)
        best_score[0] = log_initial + log_emission[0]

        # Dynamic programming finds the best predecessor for each state and time.
        for t in range(1, time_steps):
            candidates = best_score[t - 1][:, None] + log_transition
            backpointer[t] = np.argmax(candidates, axis=0)
            best_score[t] = log_emission[t] + np.max(candidates, axis=0)

        path = np.empty(time_steps, dtype=int)
        path[-1] = int(np.argmax(best_score[-1]))
        for t in range(time_steps - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]
        return path


def zscore(train, other):
    mean=np.concatenate(train).mean(axis=0); sd=np.concatenate(train).std(axis=0)
    sd[sd < 1e-9]=1.0
    return [(x-mean)/sd for x in train], [(x-mean)/sd for x in other], mean, sd


def evaluate(records, out_dir):
    train=[r for r in records if r["index"] < TRAIN_MAX_INDEX+1]
    test=[r for r in records if r["index"] >= TRAIN_MAX_INDEX+1]
    if not train or not test:
        raise ValueError("The split must contain training and held-out recordings")
    for activity in ACTIVITIES:
        if not any(r["activity"] == activity for r in train) or not any(r["activity"] == activity for r in test):
            raise ValueError(f"Both splits need at least one recording for {activity}")
    Xtrain, Xtest, mean, sd=zscore([r["X"] for r in train],[r["X"] for r in test])
    state_ids={a:i for i,a in enumerate(ACTIVITIES)}
    model=DiagonalGaussianHMM().fit(Xtrain,[state_ids[r["activity"]] for r in train])
    y_true=[]; y_pred=[]; per_record=[]
    for r,X in zip(test,Xtest):
        pred=model.viterbi(X); truth=state_ids[r["activity"]]
        y_true.extend([truth]*len(pred)); y_pred.extend(pred.tolist())
        votes=np.bincount(pred, minlength=len(ACTIVITIES))
        file_prediction=int(np.argmax(votes))
        per_record.append({"recording":r["recording"],"activity":r["activity"],"windows":len(pred),"correct":int((pred==truth).sum()),"accuracy":float((pred==truth).mean()),"predicted_activity":ACTIVITIES[file_prediction],"recording_correct":int(file_prediction==truth),"predicted":pred.tolist()})
    n=len(ACTIVITIES); cm=np.zeros((n,n),dtype=int); recording_cm=np.zeros((n,n),dtype=int)
    for t,p in zip(y_true,y_pred): cm[t,p]+=1
    for row in per_record:
        recording_cm[state_ids[row["activity"]], state_ids[row["predicted_activity"]]] += 1
    total=cm.sum(); accuracy=float(np.trace(cm)/total) if total else 0.0
    metrics=[]
    recording_metrics=[]
    recording_total=recording_cm.sum()
    for i,a in enumerate(ACTIVITIES):
        tp=cm[i,i]; fn=cm[i,:].sum()-tp; fp=cm[:,i].sum()-tp; tn=total-tp-fn-fp
        metrics.append({"activity":a,"n_test_windows":int(cm[i,:].sum()),"sensitivity":tp/(tp+fn) if tp+fn else 0.,"specificity":tn/(tn+fp) if tn+fp else 0.,"accuracy_ovr":(tp+tn)/total if total else 0.,"precision":tp/(tp+fp) if tp+fp else 0.})
        rtp=recording_cm[i,i]; rfn=recording_cm[i,:].sum()-rtp; rfp=recording_cm[:,i].sum()-rtp; rtn=recording_total-rtp-rfn-rfp
        recording_metrics.append({"activity":a,"n_test_recordings":int(recording_cm[i,:].sum()),"sensitivity":rtp/(rtp+rfn) if rtp+rfn else 0.,"specificity":rtn/(rtn+rfp) if rtn+rfp else 0.,"accuracy_ovr":(rtp+rtn)/recording_total if recording_total else 0.,"precision":rtp/(rtp+rfp) if rtp+rfp else 0.})
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    with (out/"metrics.csv").open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=metrics[0].keys()); w.writeheader(); w.writerows(metrics)
    with (out/"confusion_matrix.csv").open('w',newline='') as f:
        w=csv.writer(f); w.writerow(["true\\pred"]+ACTIVITIES)
        for a,row in zip(ACTIVITIES,cm): w.writerow([a]+row.tolist())
    with (out/"recording_metrics.csv").open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=recording_metrics[0].keys()); w.writeheader(); w.writerows(recording_metrics)
    with (out/"recording_confusion_matrix.csv").open('w',newline='') as f:
        w=csv.writer(f); w.writerow(["true/pred"]+ACTIVITIES)
        for a,row in zip(ACTIVITIES,recording_cm): w.writerow([a]+row.tolist())
    with (out/"transition_matrix.csv").open('w',newline='') as f:
        w=csv.writer(f); w.writerow(["state"]+ACTIVITIES)
        for a,row in zip(ACTIVITIES,model.A): w.writerow([a]+row.tolist())
    with (out/"initial_probabilities.csv").open('w',newline='') as f:
        w=csv.writer(f); w.writerow(["activity","probability"])
        w.writerows(zip(ACTIVITIES,model.pi.tolist()))
    with (out/"emission_parameters.csv").open('w',newline='') as f:
        w=csv.writer(f); w.writerow(["activity","feature","mean_z","variance_z"])
        for state,activity in enumerate(ACTIVITIES):
            for feature,mean,var in zip(FEATURE_NAMES,model.means[state],model.vars[state]):
                w.writerow([activity,feature,mean,var])
    with (out/"training_history.csv").open('w',newline='') as f:
        w=csv.writer(f); w.writerow(["iteration","log_likelihood"])
        w.writerows(enumerate(model.history, start=1))
    (out/"model_summary.json").write_text(json.dumps({"converged":model.converged,"stop_reason":model.stop_reason,"iterations":model.iterations,"tolerance":model.tol,"transition_prior":model.transition_prior,"final_log_likelihood":model.final_log_likelihood,"window_accuracy":accuracy,"recording_accuracy":float(np.trace(recording_cm)/recording_total) if recording_total else 0.0}, indent=2)+"\n")
    with (out/"test_recordings.csv").open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=per_record[0].keys()); w.writeheader(); w.writerows(per_record)
    save_heatmap(out/"recording_confusion_matrix.svg",recording_cm,ACTIVITIES,"Unseen test recordings confusion matrix",ACTIVITIES,integer=True)
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
    print("Baum-Welch converged:",model.converged,"reason:",model.stop_reason,"iterations:",model.iterations,"final training log likelihood:",round(model.final_log_likelihood,3))
    for row in metrics: print(row)
    print("Confusion matrix (rows=true, columns=predicted):\n",cm)
    with (Path(out_dir)/"recording_metrics.csv").open(newline="") as f:
        print("Recording-level metrics:")
        for row in csv.DictReader(f): print(row)
    print("Wrote results and SVG figures to",out_dir)

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "data", sys.argv[2] if len(sys.argv)>2 else "results")
