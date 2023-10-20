import pickle
from PhaseImages import PhaseImages

PI = pickle.load(open("../data/expt/PhaseImages_0.025ms_20230829.pickle", "rb"))
PI.plot_phase_images(6)




# # Histogram of errors
# nbins = 200
# min_error = min(min(PI.nn_mse, PI.svd_mse))
# max_error = max(max(PI.nn_mse, PI.svd_mse))
# bins = torch.linspace(min_error, max_error, nbins)
# nn_mse_histo = torch.histogram(torch.tensor(PI.nn_mse), bins=bins)
# svd_mse_histo = torch.histogram(torch.tensor(PI.svd_mse), bins=bins)
# fig, ax = plt.subplots(1, 2, figsize=(6, 2), dpi=150)
# ax[0].bar(nn_mse_histo[1][:-1], nn_mse_histo[0], width=nn_mse_histo[1][1] - nn_mse_histo[1][0], linewidth=0.1, edgecolor='black')
# ax[0].set_title("SRN3D")
# ax[0].set_yscale('log')
# ax[1].bar(svd_mse_histo[1][:-1], svd_mse_histo[0], width=svd_mse_histo[1][1] - svd_mse_histo[1][0], linewidth=0.1, edgecolor='black')
# ax[1].set_yscale('log')
# ax[1].set_title("SVD")
# dress_fig(tight=True, xlabel='MSE', ylabel='Counts')