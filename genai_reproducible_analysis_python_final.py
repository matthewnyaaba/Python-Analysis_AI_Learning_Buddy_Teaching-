"""
Reproducible analysis for:
Generative AI as a Learning Buddy and Teaching Assistant among Preservice Teachers

Inputs
------
GenAI_Reconciled_Cleaned_Data.csv

Dependencies
------------
numpy
scipy
statsmodels

The script intentionally reads CSV with Python's standard csv module so that
the analysis does not depend on Excel-reading packages.

Important
---------
The original analyses were conducted in SPSS Version 23. The revised analyses
were independently reanalysed and extended in Python using SciPy, statsmodels,
and numerical routines implemented below. The cleaned CSV restores genuine
missing values documented in the reconciliation audit.
"""

import csv, math, os, warnings
from collections import Counter
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.multivariate.factor_rotation import rotate_factors
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import jarque_bera

DATA = "GenAI_Reconciled_Cleaned_Data.csv"
OUTDIR = "analysis_outputs"
os.makedirs(OUTDIR, exist_ok=True)

def fnum(x):
    if x is None or x == "":
        return None
    return float(x)

def fint(x):
    if x is None or x == "":
        return None
    return int(float(x))

with open(DATA, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# -------------------------------
# Data helpers
# -------------------------------
for r in rows:
    r["Age code"] = fint(r["Age code"])
    r["Gender code"] = fint(r["Gender code"])
    r["Level code"] = fint(r["Level code"])
    r["Learning frequency code"] = fint(r["Learning frequency code"])
    r["Teaching frequency code"] = fint(r["Teaching frequency code"])
    r["Competence code"] = fint(r["Competence code"])
    for i in range(1, 13):
        r[f"Attitude {i}"] = fnum(r[f"Attitude {i}"])

def write_csv(path, header, body):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)

def cronbach_alpha(mat):
    mat = np.asarray(mat, float)
    k = mat.shape[1]
    item_var = mat.var(axis=0, ddof=1)
    total = mat.sum(axis=1)
    return k/(k-1) * (1 - item_var.sum()/total.var(ddof=1))

def scale_values(item_numbers):
    vals, case_ids = [], []
    for r in rows:
        x = [r[f"Attitude {i}"] for i in item_numbers]
        if all(v is not None for v in x):
            vals.append(x)
            case_ids.append(int(r["Case ID"]))
    return np.asarray(vals, float), case_ids

def scale_score(r, item_numbers):
    x = [r[f"Attitude {i}"] for i in item_numbers]
    if any(v is None for v in x):
        return None
    return float(np.mean(x))

# -------------------------------
# RQ1 task frequencies
# -------------------------------
learning_tasks = ["Reading materials","Indepth content explanations","Practical examples","Reflections","Global perspectives"]
teaching_tasks = ["Teaching Resources","Assessment strategies","Lesson Objectives","Lesson plans","Sample of lessons"]

def multiselect_count(field, labels):
    out = {x: 0 for x in labels}
    for r in rows:
        parts = [p.strip() for p in (r[field] or "").split(",")]
        for lab in labels:
            if lab in parts:
                out[lab] += 1
    return out

task_rows = []
for domain, field, labels in [
    ("Learning buddy","Learning tasks",learning_tasks),
    ("Teaching assistant","Actual teaching tasks",teaching_tasks)
]:
    counts = multiselect_count(field, labels)
    for lab in labels:
        n = counts[lab]
        task_rows.append([domain, lab, n, n/len(rows)])
write_csv(os.path.join(OUTDIR,"task_use.csv"), ["Domain","Task","n","Proportion"], task_rows)

# -------------------------------
# RQ2 paired frequencies
# -------------------------------
learning = np.array([r["Learning frequency code"] for r in rows], float)
teaching = np.array([r["Teaching frequency code"] for r in rows], float)
diff = learning - teaching
wil = stats.wilcoxon(learning, teaching, zero_method="wilcox", alternative="two-sided", correction=False, method="approx")
z_abs = stats.norm.isf(wil.pvalue/2)
# Sign follows teaching - learning, matching manuscript convention.
z = -z_abs if np.nanmean(learning - teaching) > 0 else z_abs
nonzero = int(np.sum(diff != 0))
effect_r = abs(z)/math.sqrt(nonzero)
rho, rho_p = stats.spearmanr(learning, teaching)

freq_summary = [
    ["Learning mean", learning.mean()],
    ["Learning SD", learning.std(ddof=1)],
    ["Teaching mean", teaching.mean()],
    ["Teaching SD", teaching.std(ddof=1)],
    ["Learning higher", int(np.sum(diff > 0))],
    ["Equal", int(np.sum(diff == 0))],
    ["Teaching higher", int(np.sum(diff < 0))],
    ["Wilcoxon W", wil.statistic],
    ["Wilcoxon z", z],
    ["Wilcoxon p", wil.pvalue],
    ["Non-zero pairs", nonzero],
    ["Effect size r", effect_r],
    ["Spearman rho", rho],
    ["Spearman p", rho_p],
]
write_csv(os.path.join(OUTDIR,"frequency_analysis.csv"), ["Statistic","Value"], freq_summary)

# Level 400 sensitivity analysis
lvl400 = [r for r in rows if r["Program level"] == "400"]
l4 = np.array([r["Learning frequency code"] for r in lvl400], float)
t4 = np.array([r["Teaching frequency code"] for r in lvl400], float)
d4 = l4 - t4
w4 = stats.wilcoxon(l4, t4, zero_method="wilcox", alternative="two-sided", correction=False, method="approx")
z4abs = stats.norm.isf(w4.pvalue/2)
z4 = -z4abs if np.nanmean(l4-t4) > 0 else z4abs
r4 = abs(z4)/math.sqrt(np.sum(d4 != 0))
write_csv(os.path.join(OUTDIR,"level400_sensitivity.csv"),
          ["Statistic","Value"],
          [["n",len(lvl400)],["Learning mean",l4.mean()],["Learning SD",l4.std(ddof=1)],
           ["Teaching mean",t4.mean()],["Teaching SD",t4.std(ddof=1)],
           ["Learning higher",int(np.sum(d4>0))],["Equal",int(np.sum(d4==0))],
           ["Teaching higher",int(np.sum(d4<0))],["W",w4.statistic],["z",z4],["p",w4.pvalue],
           ["non-zero pairs",int(np.sum(d4!=0))],["r",r4]])

# -------------------------------
# RQ3 descriptive attitudes and reliability
# -------------------------------
attitude_rows = []
for i in range(1,13):
    x = np.array([r[f"Attitude {i}"] for r in rows if r[f"Attitude {i}"] is not None], float)
    se = x.std(ddof=1)/math.sqrt(len(x))
    crit = stats.t.ppf(.975, len(x)-1)
    attitude_rows.append([
        i, len(x), x.mean(), x.std(ddof=1),
        x.mean()-crit*se, x.mean()+crit*se,
        np.mean(x<=2), np.mean(x==3), np.mean(x>=4)
    ])
write_csv(os.path.join(OUTDIR,"attitude_descriptives.csv"),
          ["Item","Valid n","Mean","SD","95% CI low","95% CI high","Disagree","Neutral","Agree"],
          attitude_rows)

scale_defs = {
    "Positive learning support":[1,3,4,6,8],
    "Positive teaching support":[2,5,7,11],
    "Combined positive affordances":[1,2,3,4,5,6,7,8,11],
    "Concern items":[9,10,12],
    "Original 12-item total":list(range(1,13)),
}
scale_rows = []
for name, items in scale_defs.items():
    mat, ids = scale_values(items)
    scale_rows.append([name, len(items), len(ids), np.mean(mat.mean(axis=1)), np.std(mat.mean(axis=1),ddof=1), cronbach_alpha(mat)])
write_csv(os.path.join(OUTDIR,"scale_reliability.csv"),
          ["Scale","Items","Complete n","Mean of scale","SD of scale","Cronbach alpha"], scale_rows)

# KMO and Bartlett
X12, complete_ids = scale_values(list(range(1,13)))
n,p = X12.shape
R = np.corrcoef(X12, rowvar=False)
invR = np.linalg.inv(R)
partial = np.zeros_like(R)
for i in range(p):
    for j in range(p):
        if i == j:
            partial[i,j] = 1
        else:
            partial[i,j] = -invR[i,j]/math.sqrt(invR[i,i]*invR[j,j])
r2 = sum(R[i,j]**2 for i in range(p) for j in range(i+1,p))
p2 = sum(partial[i,j]**2 for i in range(p) for j in range(i+1,p))
kmo = r2/(r2+p2)
_, logdet = np.linalg.slogdet(R)
bart_chi = -(n-1-(2*p+5)/6)*logdet
bart_df = p*(p-1)//2
bart_p = stats.chi2.sf(bart_chi,bart_df)

# Parallel analysis
rng = np.random.default_rng(2026)
n_iter = 2000
rand_eigs = np.zeros((n_iter,p))
for b in range(n_iter):
    Z = rng.normal(size=(n,p))
    rand_eigs[b] = np.linalg.eigvalsh(np.corrcoef(Z,rowvar=False))[::-1]
obs_eigs = np.linalg.eigvalsh(R)[::-1]
pa95 = np.percentile(rand_eigs,95,axis=0)
n_factors = int(np.sum(obs_eigs > pa95))

# Principal-axis factoring (iterated communalities) + oblimin/quartimin rotation
def principal_axis(R, nf, tol=1e-8, max_iter=1000):
    inv = np.linalg.inv(R)
    comm = np.clip(1 - 1/np.diag(inv), 0, 1)
    for _ in range(max_iter):
        Rred = R.copy()
        np.fill_diagonal(Rred, comm)
        vals, vecs = np.linalg.eigh(Rred)
        idx = np.argsort(vals)[::-1][:nf]
        vals = vals[idx]
        vecs = vecs[:,idx]
        load = vecs * np.sqrt(np.maximum(vals,0))
        new_comm = np.sum(load**2, axis=1)
        if np.max(np.abs(new_comm-comm)) < tol:
            return load, new_comm
        comm = new_comm
    return load, comm

unrot, communalities = principal_axis(R,n_factors)
rot, T = rotate_factors(unrot, "oblimin", 0, "oblique")
# Factor signs are arbitrary; orient to match the manuscript's positive presentation.
signs = np.ones(rot.shape[1])
for j in range(rot.shape[1]):
    if np.sum(rot[:,j]) < 0:
        rot[:,j] *= -1
        signs[j] = -1

# For the oblique solution, Phi = T' T. Apply the same sign orientation used
# for the pattern matrix so that the reported factor correlations correspond
# to the displayed loadings.
D = np.diag(signs)
factor_corr = D @ (T.T @ T) @ D
common_variance_prop = float(np.sum(communalities) / p)
common_variance_pct = 100 * common_variance_prop

efa_rows = []
for i in range(p):
    efa_rows.append([i+1, communalities[i], *rot[i,:]])
write_csv(os.path.join(OUTDIR,"efa_pattern_matrix.csv"),
          ["Item","Communality"]+[f"Factor {j+1}" for j in range(n_factors)], efa_rows)
write_csv(os.path.join(OUTDIR,"efa_diagnostics.csv"),
          ["Statistic","Value"],
          [["Complete n",n],["KMO",kmo],["Bartlett chi-square",bart_chi],["Bartlett df",bart_df],["Bartlett p",bart_p],
           ["Factors retained by parallel analysis",n_factors],
           ["Common variance accounted for by retained factors (proportion)",common_variance_prop],
           ["Common variance accounted for by retained factors (percent)",common_variance_pct]])
write_csv(os.path.join(OUTDIR,"efa_factor_correlations.csv"),
          ["Factor 1","Factor 2","Correlation"],
          [[i+1,j+1,factor_corr[i,j]] for i in range(n_factors) for j in range(i+1,n_factors)])
write_csv(os.path.join(OUTDIR,"parallel_analysis.csv"),
          ["Component","Observed eigenvalue","95th percentile random eigenvalue"],
          [[i+1,obs_eigs[i],pa95[i]] for i in range(p)])

# -------------------------------
# RQ4 regression helpers
# -------------------------------
def design_row(r, include_skill=False, include_institution=False):
    if r["Age"] == "":
        return None
    age = r["Age"]
    gender = r["Gender"]
    level = r["Program level"]
    x = [
        1 if age=="16-20" else 0,
        1 if age=="26-30" else 0,
        1 if age=="Above 31" else 0,
        1 if gender=="Female" else 0,
        1 if level=="200" else 0,
        1 if level=="300" else 0,
    ]
    names = ["Age 16-20","Age 26-30","Age above 31","Female","Level 200","Level 300"]
    if include_skill:
        x.append(r["Competence code"]); names.append("Self-rated GenAI competence")
    if include_institution:
        x.extend([
            1 if r["Institution"]=="Gambaga College of Education" else 0,
            1 if r["Institution"]=="University for Development Studies" else 0,
            1 if r["Institution"]=="St. Joseph's College of Education" else 0,
        ])
        names += ["Institution: Gambaga","Institution: UDS","Institution: St Joseph"]
    return x,names

def fit_linear(item_nums, outcome_name, include_skill=False, include_institution=False):
    y,x,case_ids,names = [],[],[],None
    for r in rows:
        score = scale_score(r,item_nums)
        d = design_row(r,include_skill,include_institution)
        if score is None or d is None:
            continue
        rowx,names = d
        y.append(score); x.append(rowx); case_ids.append(int(r["Case ID"]))
    X = sm.add_constant(np.asarray(x,float),has_constant="add")
    # Fit the ordinary OLS model for residual/influence diagnostics, then use
    # HC3 covariance for coefficient inference and the robust omnibus F test.
    base_model = sm.OLS(np.asarray(y,float),X).fit()
    # Keep the same HC3 fitting call used in the reconciled analysis so the
    # published coefficients, p-values, and confidence intervals are unchanged.
    model = sm.OLS(np.asarray(y,float),X).fit(cov_type="HC3")
    out_names = ["Intercept"]+names
    coef = []
    ci = model.conf_int()
    for name,b,se,pv,lo,hi in zip(out_names,model.params,model.bse,model.pvalues,
                                  ci[:,0],ci[:,1]):
        coef.append([outcome_name,name,b,se,lo,hi,pv])
    vifs = [[outcome_name,nm,variance_inflation_factor(X,i+1)]
            for i,nm in enumerate(names)]
    bp = het_breuschpagan(base_model.resid,X)
    jb = jarque_bera(base_model.resid)
    reset = linear_reset(base_model, power=2, use_f=True)
    cooks_d = OLSInfluence(base_model).cooks_distance[0]
    diag = [
        [outcome_name,"n",len(y)],
        [outcome_name,"R2",base_model.rsquared],
        [outcome_name,"Adjusted R2",base_model.rsquared_adj],
        [outcome_name,"Robust F",float(model.fvalue)],
        [outcome_name,"Model p",float(model.f_pvalue)],
        [outcome_name,"Max VIF",max(v[2] for v in vifs)],
        [outcome_name,"Breusch-Pagan chi-square",bp[0]],
        [outcome_name,"Breusch-Pagan p",bp[1]],
        [outcome_name,"Jarque-Bera chi-square",jb[0]],
        [outcome_name,"Jarque-Bera p",jb[1]],
        [outcome_name,"Ramsey RESET F",float(reset.fvalue)],
        [outcome_name,"Ramsey RESET p",float(reset.pvalue)],
        [outcome_name,"Max Cooks D",float(np.max(cooks_d))],
    ]
    return coef,vifs,diag

linear_coef, vif_rows, linear_diag = [],[],[]
for items,name in [
    ([1,3,4,6,8],"Positive learning-support attitude"),
    ([2,5,7,11],"Positive teaching-support attitude")
]:
    for skill in (False,True):
        label = name + (" + competence" if skill else "")
        c,v,d = fit_linear(items,label,include_skill=skill)
        linear_coef += c; vif_rows += v; linear_diag += d
write_csv(os.path.join(OUTDIR,"linear_regression_coefficients.csv"),
          ["Outcome/model","Predictor","B","HC3 SE","95% CI low","95% CI high","p"], linear_coef)
write_csv(os.path.join(OUTDIR,"linear_regression_vif.csv"),
          ["Outcome/model","Predictor","VIF"], vif_rows)
write_csv(os.path.join(OUTDIR,"linear_regression_diagnostics.csv"),
          ["Outcome/model","Diagnostic","Value"], linear_diag)

def fit_ordinal(outcome_col, outcome_name, include_skill=False, include_institution=False):
    y,x,names = [],[],None
    for r in rows:
        d=design_row(r,include_skill,include_institution)
        if d is None:
            continue
        rowx,names=d
        y.append(r[outcome_col]); x.append(rowx)
    y=np.asarray(y,int); X=np.asarray(x,float)
    model=OrderedModel(y,X,distr="logit")
    res=model.fit(method="bfgs",disp=False,maxiter=1000)
    q=len(names)
    ci=res.conf_int()
    coef=[]
    for i,nm in enumerate(names):
        b=res.params[i]
        coef.append([outcome_name,nm,b,res.bse[i],math.exp(b),math.exp(ci[i,0]),math.exp(ci[i,1]),res.pvalues[i]])
    lr=2*(res.llf-res.llnull)
    fitrow=[outcome_name,len(y),res.llf,lr,q,stats.chi2.sf(lr,q),1-res.llf/res.llnull,res.aic,res.bic]
    return coef,fitrow,y,X,names

ordinal_coef, ordinal_fit = [],[]
ordinal_models={}
for col,name in [
    ("Learning frequency code","Learning-buddy use frequency"),
    ("Teaching frequency code","Teaching-assistant use frequency")
]:
    for skill in (False,True):
        label=name + (" + competence" if skill else "")
        c,f,y,X,names=fit_ordinal(col,label,skill)
        ordinal_coef += c; ordinal_fit.append(f)
        ordinal_models[label]=(y,X,names)
write_csv(os.path.join(OUTDIR,"ordinal_regression_coefficients.csv"),
          ["Outcome/model","Predictor","B","SE","OR","95% CI low","95% CI high","p"], ordinal_coef)
write_csv(os.path.join(OUTDIR,"ordinal_regression_fit.csv"),
          ["Outcome/model","n","Log likelihood","LR chi-square","df","p","McFadden R2","AIC","BIC"], ordinal_fit)

# -------------------------------
# Proportional-odds diagnostics
# -------------------------------
# Standard Brant (1990)-type Wald test implemented from the four nested
# cumulative binary logistic regressions. Cross-threshold covariance is
# retained, which distinguishes this from simply comparing independent binary
# models. The per-predictor test has J-2 = 3 df for a five-category outcome.
def brant_test(y, X, names):
    Z = sm.add_constant(np.asarray(X,float),has_constant="add")
    all_names = ["Intercept"] + list(names)
    cuts = range(1,5)
    fits, probs, covs = [], [], []
    for cut in cuts:
        yb = (np.asarray(y) > cut).astype(int)
        fit = sm.GLM(yb,Z,family=sm.families.Binomial()).fit(maxiter=1000)
        fits.append(fit)
        probs.append(np.asarray(fit.fittedvalues))
        covs.append(np.asarray(fit.cov_params()))

    K = len(fits)
    q = Z.shape[1]
    beta = np.concatenate([np.asarray(f.params) for f in fits])
    V = np.zeros((K*q,K*q))
    for j in range(K):
        for k in range(K):
            pj, pk = probs[j], probs[k]
            if j == k:
                c = pj * (1-pj)
            elif j < k:
                # For Y>cut binary outcomes, the higher cut is nested inside
                # the lower cut, so P(Y_j=1,Y_k=1)=P(Y_k=1).
                c = pk * (1-pj)
            else:
                c = pj * (1-pk)
            middle = Z.T @ (c[:,None] * Z)
            Vjk = covs[j] @ middle @ covs[k]
            V[j*q:(j+1)*q,k*q:(k+1)*q] = Vjk

    rows_out = []
    # Predictor-specific tests, excluding intercept from the reported table.
    for m,name in enumerate(all_names[1:], start=1):
        C = np.zeros((K-1,K*q))
        for r in range(1,K):
            C[r-1,r*q+m] = 1
            C[r-1,m] = -1
        d = C @ beta
        S = C @ V @ C.T
        chi2 = float(d.T @ np.linalg.pinv(S) @ d)
        df_test = K-1
        rows_out.append([name,chi2,df_test,float(stats.chi2.sf(chi2,df_test))])

    # Omnibus test across all slopes (intercepts/cutpoints excluded).
    Crows = []
    for m in range(1,q):
        for r in range(1,K):
            c = np.zeros(K*q)
            c[r*q+m] = 1
            c[m] = -1
            Crows.append(c)
    C = np.vstack(Crows)
    d = C @ beta
    S = C @ V @ C.T
    chi2 = float(d.T @ np.linalg.pinv(S) @ d)
    omnibus = ["Omnibus",chi2,C.shape[0],float(stats.chi2.sf(chi2,C.shape[0]))]
    return rows_out, omnibus

brant_rows=[]
po_rows=[]
for label,(y,X,names) in ordinal_models.items():
    # Brant tests are especially relevant to the final competence-adjusted
    # models but are generated for every fitted ordinal model for transparency.
    per_pred, omnibus = brant_test(y,X,names)
    brant_rows.append([label,*omnibus])
    for r in per_pred:
        brant_rows.append([label,*r])

    # Threshold-specific binary logits show the direction and size of effects
    # at each cumulative cut-point. Sparse cells are flagged rather than hidden.
    Z=sm.add_constant(X,has_constant="add")
    for cut in range(1,5):
        yb=(y>cut).astype(int)
        fit=sm.GLM(yb,Z,family=sm.families.Binomial()).fit(maxiter=500)
        ci=fit.conf_int()
        for i,nm in enumerate(names, start=1):
            flag = "SPARSE/UNSTABLE" if abs(fit.params[i])>8 or fit.bse[i]>10 else ""
            exp_safe = lambda v: float(np.exp(np.clip(v, -700, 700)))
            po_rows.append([label,cut,nm,fit.params[i],fit.bse[i],
                            exp_safe(fit.params[i]),exp_safe(ci[i,0]),exp_safe(ci[i,1]),
                            fit.pvalues[i],flag])
write_csv(os.path.join(OUTDIR,"proportional_odds_brant.csv"),
          ["Outcome/model","Test/Predictor","Chi-square","df","p"],brant_rows)
write_csv(os.path.join(OUTDIR,"proportional_odds_threshold_diagnostic.csv"),
          ["Outcome/model","Cut point: Y >","Predictor","Threshold-specific B","SE","OR","95% CI low","95% CI high","p","Flag"],po_rows)

# -------------------------------
# Institution-adjusted sensitivity models
# -------------------------------
inst_rows=[]
inst_block_rows=[]
for items,name in [
    ([1,3,4,6,8],"Learning attitude + competence + institution"),
    ([2,5,7,11],"Teaching attitude + competence + institution")
]:
    c,_,d=fit_linear(items,name,include_skill=True,include_institution=True)
    inst_rows += [[row[0],row[1],row[2],row[-1]] for row in c]

    # Robust Wald chi-square test of the three institution indicators as a block.
    y,x,names = [],[],None
    for r in rows:
        score = scale_score(r,items)
        des = design_row(r,True,True)
        if score is None or des is None:
            continue
        rowx,names = des
        y.append(score); x.append(rowx)
    Xfull = sm.add_constant(np.asarray(x,float),has_constant="add")
    robust = sm.OLS(np.asarray(y,float),Xfull).fit(cov_type="HC3")
    Rtest = np.zeros((3,len(robust.params)))
    Rtest[0,-3] = 1; Rtest[1,-2] = 1; Rtest[2,-1] = 1
    wt = robust.wald_test(Rtest,use_f=False,scalar=True)
    inst_block_rows.append([name,"Robust Wald chi-square",float(wt.statistic),3,float(wt.pvalue)])

for col,name in [
    ("Learning frequency code","Learning use + competence + institution"),
    ("Teaching frequency code","Teaching use + competence + institution")
]:
    c,f,y_full,X_full,names_full=fit_ordinal(col,name,include_skill=True,include_institution=True)
    inst_rows += [[row[0],row[1],row[2],row[-1]] for row in c]
    _,f_reduced,_,_,_=fit_ordinal(col,name.replace(" + institution",""),include_skill=True,include_institution=False)
    lr = 2 * (f[2] - f_reduced[2])
    inst_block_rows.append([name,"Likelihood-ratio chi-square",lr,3,float(stats.chi2.sf(lr,3))])

write_csv(os.path.join(OUTDIR,"institution_sensitivity.csv"),
          ["Outcome/model","Predictor","Coefficient B","p"],inst_rows)
write_csv(os.path.join(OUTDIR,"institution_block_tests.csv"),
          ["Outcome/model","Test","Statistic","df","p"],inst_block_rows)

print("Analysis complete. Outputs written to:", OUTDIR)
print("KMO =", round(kmo,3), "; Bartlett chi-square =", round(bart_chi,2), "; factors retained =", n_factors)
print("Retained-factor common variance =", round(common_variance_pct,2), "% ; Factor 1-2 correlation =", round(factor_corr[0,1],3))
print("Wilcoxon W =", wil.statistic, "; z =", round(z,3), "; p =", wil.pvalue, "; r =", round(effect_r,3))
