from hoh import *

def state_draw(s, sp, a=None, who=None, howmuch=1 / 5, outcome=0, fname='out.png', idx=None):
    scene_fname = f'{fname}.pov'
    with open(scene_fname, 'w') as f:
        f.write('''
        #include "colors.inc"
        #include "stones.inc"
        #include "woods.inc"
        #include "metals.inc"
        #include "stones.inc"
        #include "stones2.inc"
        #include "finish.inc"

//        global_settings { radiosity { } }

        light_source {
            <4.5, 60, 0.1>
            color White
            area_light <10, 0, 0>, <0, 0, 10>, 5, 5
        }

        camera {
            location <5.5,14,-5.5>
            look_at  <5.5,0,3.5>
            angle 55
        }

        #declare T_cage = texture {
          T_Chrome_1A
          scale 1
        }
        ''')
        
        # DRAW A
        f.write('''
            difference {
                box {
                    0, 1
                    texture { pigment { color Yellow } }
                }
        ''')
        
        import hashlib
        for n in range(50):
            slug = hashlib.md5(str(n).encode('utf-8')).digest()
            x = slug[0] / 256 * 1
            y = slug[1] / 256 * 1
            z = slug[2] / 256 * 1
            r = slug[2] / 256 * 0.1 + 0.1
            f.write(f'''
                sphere {{
                    <{x}, {y}, {z}>, {r}
                    texture {{ pigment {{ color Yellow }} }}
                }}
            ''')

#         f.write('''
#         difference {
#             box {
#                 0, 1
#                 texture { T_cage }
#             }
#             box {
#                 0.05, 0.95
#                 texture { T_cage }
#             }
#         ''')

#         for x,y in itertools.product(range(5), range(5)):
#             margin = 0.05
#             x1 = margin + (1 - margin)/5 * x
#             y1 = margin + (1 - margin)/5 * y
#             x2 = x1 + (1 - margin)/5 - margin
#             y2 = y1 + (1 - margin)/5 - margin
#             f.write('''
#             box {
#                 <-0.1, X1, Y1>,
#                 <+1.1, X2, Y2>
#                 texture { T_cage }
#             }
#             box {
#                 <X1, -0.1, Y1>,
#                 <X2, +1.1, Y2>
#                 texture { T_cage }
#             }
#             box {
#                 <X1, Y1, -0.1>,
#                 <X2, Y2, +1.1>
#                 texture { T_cage }
#             }
#         '''.replace('X1', str(x1)).replace('X2', str(x2)).replace('Y1', str(y1)).replace('Y2', str(y2)))

        ax, ay = s.ax, s.ay
        if who == 'b' and a != Actions.STAY and find(s, s.bx + ds[a][0], s.by + ds[a][1]) == 'a':
            ax += ds[a][0] * howmuch
            ay += ds[a][1] * howmuch
        if who == 'a' and a != Actions.STAY:
            # ax += ds[a][0] * howmuch
            # ay += ds[a][1] * howmuch
            phi = 90 * (0.5 - 2 * (howmuch - 0.5)**2 * (-1 if outcome == 0 and howmuch > 0.5 else +1))
            if a == Actions.SOUTH:
                f.write(f'''
                    translate <-1, 0, 0>
                    rotate z*{-phi}
                    translate <+1, 0, 0>
                ''')
            elif a == Actions.NORTH:
                f.write(f'''
                    rotate z*{phi}
                ''')
            elif a == Actions.EAST:
                f.write(f'''
                    translate <0, 0, -1>
                    rotate x*{+phi}
                    translate <0, 0, +1>
                ''')
            elif a == Actions.WEST:
                f.write(f'''
                    rotate x*{-phi}
                ''')
        f.write(f'''
            translate <{ax}, 0, {ay}>
        }}
        ''')

        # DRAW MAZE
        for y in range(-1, maze_h + 1):
            for x in range(-1, maze_w + 1):
                if 0 <= x < maze_w and 0 <= y < maze_h and maze[y][x] == '.':
                    pass
                else:
                    f.write(f'''
                    box {{
                        <{x}, 0, {y}>,
                        <{x+1}, 0.9, {y+1}>
                        texture {{ T_Wood7 }}
                    }}
                    box {{
                        <{x-0.05}, 0.9, {y-0.05}>,
                        <{x+1.05}, 1, {y+1.05}>
                        texture {{ T_Stone25 }}
                    }}
                    ''')
        # http://compsci.world.coocan.jp/OUJ/povtl/stones/index.html

        # Draw B
        bx, by = s.bx, s.by
        if who == 'b' and a != Actions.STAY:
            bx += ds[a][0] * howmuch
            by += ds[a][1] * howmuch
        f.write(f'''
        #declare Make_B = union {{
            cone {{
                <0.5, 0.0, 0.5>, 0.5
                <0.5, 1.5, 0.5>, 0.3
                texture {{ T_Chrome_1A }}
            }}
            sphere {{
                <0.5, 1.5, 0.5>, 0.3
                pigment {{ color Blue }}
            }}
            cylinder {{
                <0.5, 1.5, 0.5>,
                <0.5, 2.5, 0.5>,
                0.05
                texture {{ T_Chrome_1A }}
            }}
            sphere {{
                <0.5, 2.5, 0.5>, 0.1
                pigment {{ color {'Yellow' if (((idx + howmuch) / 4) % 1) < 0.25 else 'Blue'} }}
            }}
        }}
        object {{
            Make_B
            translate <{bx}, 0, {by}>
        }}
        ''')
        
        # Draw ROCK
        rx, ry = s.rx, s.ry
        if who == 'b' and a != Actions.STAY and find(s, s.bx + ds[a][0], s.by + ds[a][1]) == 'r':
            rx += ds[a][0] * howmuch
            ry += ds[a][1] * howmuch
        rz = 0
        if who == 'r':
            rx = sp.rx
            ry = sp.ry
            rz = 6 * (1 - howmuch ** 2)
        f.write(f'''
        cone {{
            <0.5, 0.0, 0.5>, 0.5
            <0.5, 0.8, 0.5>, 0.4
            texture {{ T_Wood7 }}
            translate <{rx}, {rz}, {ry}>
        }}
        ''')

        # Draw floor and goals
        f.write('''
        intersection {
            plane {
                <0,1,0>, // normal vector
                0 // distance from origin
                texture { T_Stone8 }
                // pigment { color White }
                // pigment {
                    color Gray
                //    checker color White, color Gray
                //}
            }
            box {
                <-1, -1, -1>,
                <12, +1, 8>
            }
        }

        box {
            <0.9, 0.00, 0.9>,
            <2.1, 0.01, 2.1>
            pigment { color Magenta * __T1__ }
            finish { emission 0.1 }
        }
        box {
            <0.9, 0.00, 4.9>,
            <2.1, 0.01, 6.1>
            pigment { color Green * __T2__ }
            finish { emission 0.1 }
        }
        '''.replace('__T1__',  str(np.cos((idx + howmuch) * 3) / 5 + 0.8)).replace('__T2__',  str(np.sin((idx + howmuch) * 3) / 5 + 0.8)))

    os.system(f'povray -D -I{scene_fname} -O{fname}.png -GA +W800 +H600 +Q9 +WT1 +UA 2> /dev/null')

if __name__ == '__main__':
    # import random
    # random.seed(9)
    # while True:
    #     s = state_t(*random.choice(S))
    #     if state_valid(s):
    #         break
    s = state_t(ax=5, ay=1, bx=6, by=1, rx=7, ry=1)
    state_draw(s, s, a=Actions.NORTH, who='a', howmuch=0/5, outcome=0, fname='out-0.png', idx=0)
    # state_draw(s, s, a=Actions.NORTH, who='a', howmuch=1/5, outcome=0, fname='out-1.png', idx=0)
    # state_draw(s, s, a=Actions.NORTH, who='a', howmuch=2/5, outcome=0, fname='out-2.png', idx=0)
    # state_draw(s, s, a=Actions.NORTH, who='a', howmuch=3/5, outcome=0, fname='out-3.png', idx=0)
    # state_draw(s, s, a=Actions.NORTH, who='a', howmuch=4/5, outcome=0, fname='out-4.png', idx=0)
    # state_draw(s, s, a=Actions.NORTH, who='a', howmuch=5/5, outcome=0, fname='out-5.png', idx=0)
