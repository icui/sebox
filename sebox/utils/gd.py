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
        def getmf(d, pre=False):
            mf = 0.0

            for j in range(len(d.ls())):
                print(d.path(), d.has(f'kl_{j:02d}'))
                if d.has(kl := f'kl_{j:02d}'):
                    if d.has(m := f'{kl}/phase_mf.npy'):
                        mf += d.load(m).sum()

                    if d.has(m := f'{kl}/amp_mf.npy'):
                        mf += d.load(m).sum()
                    
                else:
                    break

            if mf > 0.0:
                if pre:
                    print(d.path())

                print(f' {mf:.3e}')
                return True
            
            return False


        for i in range(len(root.ls())):
            d_i = root.subdir(f'iter_{i:02d}')

            if getmf(d_i, True):
                for k in range(len(d_i.ls())):
                    if not getmf(d_i.subdir(f'step_{k:02d}')):
                        break
            
            else:
                break

