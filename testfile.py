    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return (mean, mean <= thresh)
