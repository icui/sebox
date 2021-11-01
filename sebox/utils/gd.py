if __name__ == '__main__':
    # check misfit values
    from sebox import root
    root.restore()

    for optim in root:
        if not optim.has('misfit.npy'):
            continue
        
        print(f'Iteration {optim.iteration}')

        steps = [0.0]
        vals = [optim.load('misfit.npy').sum()]

        for step in optim[-2]:
            if not step.has('misfit.npy'):
                continue

            steps.append(step.load('step.pickle'))
            vals.append(step.load('misfit.npy').sum())
            
        for step, val in zip(steps, vals):
            if val is None:
                break
            
            entry = f' {step:.3e}: {val:.3e}'

            if optim.done and val == min(vals):
                entry += ' *'
            
            print(entry)
        
        print()
