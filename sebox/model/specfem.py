from sebox import Directory


def probe_smoother(d: Directory, hess: bool, ntotal: int):
    """Prober of smoother progress."""
    kind = 'smooth_' + ('hess' if hess else 'kl')

    if ntotal and d.has(out := f'OUTPUT_FILES/{kind}.txt'):
        n = 0

        lines = d.readlines(out)
        niter = '0'

        for line in lines:
            if 'Initial residual:' in line:
                n += 1
            
            elif 'Iterations' in line:
                niter = line.split()[1]
        
        n = max(1, n)

        return f'{n}/{ntotal*2} iter{niter}'
