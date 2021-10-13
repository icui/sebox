if __name__ == '__main__':
    # check misfit values
    from sebox import root

    try:
        root.restore()

        for i, optim in enumerate(root):
            if len(optim) < 2 or optim[0].misfit_value is None:
                continue

            print(f'Iteration {i}')
        
            steps = [0.0]
            vals = [optim[0].misfit_value]

            if len(optim):
                for step in optim[-2]:
                    steps.append(step.step)
                    vals.append(step[1].misfit_value)
            
            for step, val in zip(steps, vals):
                if val is None:
                    break
                
                entry = f' {step:.3e}: {val:.3e}'

                if optim.done and val == min(vals):
                    entry += ' *'
                
                print(entry)
            
            print()
    
    except:
        for i in range(len(root.ls())):
            if root.has(cwd := f'iter_{i:02d}'):
                print(cwd)
                mf = 0.0

                for j in range(len(root.ls(cwd))):
                    if root.has(kl := f'cwd/kl_{j:02d}'):
                        if root.has(mf := f'{kl}/phase_mf.npy'):
                            mf += root.load(mf).sum()

                        if root.has(mf := f'{kl}/amp_mf.npy'):
                            mf += root.load(mf).sum()

                if mf > 0.0:
                    print('', mf)

