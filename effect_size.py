def cohens_d(mean1, std1, mean2, std2):
    pooled_std = ((std1**2 + std2**2) / 2) ** 0.5
    return (mean1 - mean2) / pooled_std

# 5-seed ablation results (coef, mean reward, std reward)
baseline = (230.9, 14.7)
results = {
    0.5: (235.4, 9.9),
    2.0: (223.8, 25.9),
    5.0: (183.9, 4.9),
    10.0: (162.4, 27.7),
}

print(f"{'Coef':<8}{'Cohens d':<12}{'Interpretation':<15}")
for coef, (mean, std) in results.items():
    d = cohens_d(mean, std, baseline[0], baseline[1])
    mag = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
    print(f"{coef:<8}{d:<12.2f}{mag:<15}")
