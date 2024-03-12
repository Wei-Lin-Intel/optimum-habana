import matplotlib.pyplot as plt

def get_loss(fl):
    with open(fl) as h:
        lines = h.readlines()
    filt_lines = [float(ln.split('step_loss=')[-1].split(']')[0]) for ln in lines if 'step_loss=' in ln]
    return filt_lines

def compare_loss(f0, f1):
    l0 = get_loss(f0)
    l1 = get_loss(f1)
    plt.plot(l0)
    plt.plot(l1)
    plt.savefig('foo.png')
    import pdb; pdb.set_trace()
    print()


compare_loss('log_1x_r512_det_nomedpipe.txt', 'log_1x_r512_det_medpipe.txt')