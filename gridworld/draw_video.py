from hoh import *
from draw_pov import state_draw as draw_pov

def show_video(transitions, fname='out', mode='pov', verbose=True):
    state_draw = {
        'pov': draw_pov
    }[mode]
    
    os.makedirs(f'out/{fname}/imgs/', exist_ok=True)
    os.system(f'rm out/{fname}/imgs/*.png')
    i = 0
    j = 0
    
    num_subdivisions = 9
    with Pool(max(1, len(os.sched_getaffinity(0)) - 1)) as pool:
        with tqdm(total=num_subdivisions * len(transitions) + 1) as pbar:
            for s, a, who, s_ in transitions:
                # if a == Actions.STAY:
                #     continue
                j += 1
                i += 1
                for howmuch in np.linspace(0, 1, num_subdivisions, endpoint=False):
                    i += 1
                    outcome = 1 if who == 'a' and (s.ax, s.ay) == (s_.ax, s_.ay) else 0

                    if False: #mode == 'pov':
                        state_draw(s, s_, a, who, howmuch, outcome, fname=f'out/{i:05}.png', idx=j)
                    else:
                        pool.apply_async(state_draw, (s, s_, a, who, howmuch, outcome), dict(fname=f'out/{fname}/imgs/{i:05}', idx=j), callback=lambda _: pbar.update())
        
            pool.apply_async(state_draw, (s_, s_), dict(fname=f'out/{fname}/imgs/{i:05}', idx=j), callback=lambda _: pbar.update())
            pool.close()
            pool.join()

    # state_draw(s_, s_, fname=f'out/{i:05}.png', idx=j)
    os.system(f'''ffmpeg -hide_banner -loglevel error -framerate 30 -y -pattern_type glob -i 'out/{fname}/imgs/*.png' -c:v libx264 -pix_fmt yuv420p out/{fname}/{fname}.mp4''')
    #return Video(f'{fname}.mp4', html_attributes='controls autoplay')
