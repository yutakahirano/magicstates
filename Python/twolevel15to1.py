import numpy as np
from scipy import optimize
from definitions import (z, one, projx, kronecker_product, apply_rot, plog,
                         storage_x_5, storage_z_5, init5qubit, ideal15to1,
                         apply_pauli)
from onelevel15to1 import one_level_15to1_state
from enum import Enum

# Calculates the output error and cost of the (15-to-1)x(15-to-1) protocol
# with a physical error rate pphys, level-1 distances dx, dz and dm,
# level-2 distances dx2, dz2 and dm2, using nl1 level-1 factories
def cost_of_two_level_15to1_with_l0_distillation(pphys, dx2, dz2, dm2, pfail0, perror0):
    # Introduce shorthand notation for logical error rate with distances dx2/dz2/dm2
    px2 = plog(pphys, dx2)
    pz2 = plog(pphys, dz2)
    pm2 = plog(pphys, dm2)

    pfail = pfail0
    pl1 = perror0

    # Compute l1time, the speed at which level-2 rotations can be performed (t_{L1} in the paper)
    l1time = dm2
    
    # Define lmove, the effective width-dm2 region a level-1 state needs to traverse
    # before reaching the level-2 block, picking up additional storage errors
    lmove = 0

    # Step 1 of (15-to-1)x(15-to-1) protocol applying rotations 1-2
    out2 = apply_rot(init5qubit, [one, z, one, one, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [one, one, z, one, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (3 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time, 0, 0)
    out2 = storage_z_5(out2, 0, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time, 0, 0)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Step 2: apply rotations 3-4
    out2 = apply_rot(out2, [one, one, one, z, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 3 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [one, one, one, one, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dz2 + dm2) * dx2 / dm2 * pm2,
                     0)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Step 3: apply rotations 5-6
    # Last operation: apply additional storage errors due to multi-patch measurements
    out2 = apply_rot(out2, [z, z, z, one, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 2 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [one, z, z, z, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = storage_z_5(out2, 0.5 * (dx2 + 2 * dz2 + dm2) * dm2 / dx2 * px2, 0, 0, 0, 0)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0.5 * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0.5 * px2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Step 4: apply rotations 7-8
    out2 = apply_rot(out2, [z, one, z, z, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 3 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [z, z, one, z, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = storage_z_5(out2, 0.5 * ((dx2 + 3 * dz2 + dm2) + (dx2 + 4 * dz2 + dm2)) * dm2 / dx2 * px2, 0, 0, 0, 0)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0.5 * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0.5 * px2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Step 5: apply rotations 9-10
    out2 = apply_rot(out2, [z, z, one, one, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [z, one, one, z, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = storage_z_5(out2, 0.5 * ((dx2 + 4 * dz2 + dm2) + (dx2 + 4 * dz2 + dm2)) * dm2 / dx2 * px2, 0, 0, 0, 0)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0.5 * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0.5 * px2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Step 6: apply rotations 11-12
    out2 = apply_rot(out2, [z, one, z, one, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [z, z, z, z, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = storage_z_5(out2, 0.5 * ((dx2 + 4 * dz2 + dm2) + (dx2 + 4 * dz2 + dm2)) * dm2 / dx2 * px2, 0, 0, 0, 0)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0.5 * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0.5 * px2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time)

    # Step 7: apply rotations 13-14
    out2 = apply_rot(out2, [one, z, one, z, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [one, one, z, z, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (3 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = storage_z_5(out2, 0.5 * (dx2 + 4 * dz2 + dm2) * dm2 / dx2 * px2, 0, 0, 0, 0)

    # Apply storage errors for l1time code cycles
    # Qubit 1 is consumed as an output state: additional storage errors for dx2 code cycles
    out2 = storage_x_5(out2, 0.5 * px2 * (dm2 + 2 * dx2), 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0.5 * px2 * (dm2 + 2 * dx2), 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Step 8: apply rotation 15
    out2 = apply_rot(out2, [one, z, z, one, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time, 0,
                       0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time, 0,
                       0.5 * (dx2 / dz2) * pz2 * l1time)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Compute level-2 failure probability as the probability to measure qubits 2-5 in the |+> state
    pfail2 = np.real(1 - np.trace(np.dot(kronecker_product([one, projx, projx, projx, projx]), out2)))

    # Compute the density matrix of the post-selected output state, i.e., after projecting qubits 2-5 into |+>
    outpostsel2 = 1 / (1 - pfail2) * np.dot(np.dot(kronecker_product([one, projx, projx, projx, projx]), out2),
                                            kronecker_product([one, projx, projx, projx, projx]).conj().transpose())

    # Compute level-2 output error from the infidelity between the post-selected state and the ideal output state
    pout = np.real(1 - np.trace(np.dot(outpostsel2, ideal15to1)))

    # Full-distance computation: determine full distance required for a 100-qubit / 10000-qubit computation
    def logerr1(d):
        return 231 / float(pout) * d * float(plog(pphys, d[0]).real) - 0.01

    def logerr2(d):
        return 20284 / float(pout) * d * float(plog(pphys, d[0]).real) - 0.01

    reqdist1 = int(2 * round(optimize.root(logerr1, 3, method='hybr').x[0] / 2) + 1)
    reqdist2 = int(2 * round(optimize.root(logerr2, 3, method='hybr').x[0] / 2) + 1)

    # Print output error, failure probability, space cost, time cost and space-time cost
    nqubits = 2 * ((dx2 + 4 * dz2) * 3 * dx2 + 16 * dm2 * dm2 + 2 * dx2 * dm2)
    ncycles = 7.5 * l1time / (1 - float(pfail2))
    print('(15-to-1)x(15-to-1) with pphys=', pphys, ', pfail0=', pfail0, ', perror0=', perror0, ', ltime=', l1time, ', dx2=', dx2, ', dz2=', dz2,
          ', dm2=', dm2, sep='')
    print('Output error: ', '%.4g' % pout, sep='')
    print('Failure probability: ', '%.3g' % pfail2, sep='')
    print('Qubits: ', '%.0f' % nqubits, sep='')
    print('Code cycles: ', '%.2f' % ncycles, sep='')
    print('Space-time cost: ', '%.0f' % (nqubits * ncycles), ' qubitcycles', sep='')
    print('For a 100-qubit computation: ', ('%.3f' % (nqubits * ncycles / 2 / reqdist1 ** 3)), 'd^3 (d=', reqdist1, ')',
          sep='')
    print('For a 5000-qubit computation: ', ('%.3f' % (nqubits * ncycles / 2 / reqdist2 ** 3)), 'd^3 (d=', reqdist2,
          ')', sep='')
    print('')


# Calculates the output error and cost of the (15-to-1)x(15-to-1) protocol
# with a physical error rate pphys, level-1 distances dx, dz and dm,
# level-2 distances dx2, dz2 and dm2, using nl1 level-1 factories
def cost_of_two_level_15to1(pphys, dx, dz, dm, dx2, dz2, dm2, nl1):
    # Introduce shorthand notation for logical error rate with distances dx2/dz2/dm2
    px2 = plog(pphys, dx2)
    pz2 = plog(pphys, dz2)
    pm2 = plog(pphys, dm2)

    # Compute pl1, the output error of level-1 states
    out = one_level_15to1_state(pphys, dx, dz, dm)
    pfail = float(np.real(1 - np.trace(np.dot(kronecker_product([one, projx, projx, projx, projx]), out))))
    outpostsel = 1 / (1 - pfail) * np.dot(np.dot(kronecker_product([one, projx, projx, projx, projx]), out),
                                          kronecker_product([one, projx, projx, projx, projx]).conj().transpose())
    pl1 = float(np.real(1 - np.trace(np.dot(outpostsel, ideal15to1))))

    # Compute l1time, the speed at which level-2 rotations can be performed (t_{L1} in the paper)
    l1time = max(6 * dm / (nl1 / 2) / (1 - pfail), dm2)

    # Define lmove, the effective width-dm2 region a level-1 state needs to traverse
    # before reaching the level-2 block, picking up additional storage errors
    lmove = 10 * dm2 + nl1 / 4 * (dx + 4 * dz)

    # Step 1 of (15-to-1)x(15-to-1) protocol applying rotations 1-2
    out2 = apply_rot(init5qubit, [one, z, one, one, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [one, one, z, one, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (3 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time, 0, 0)
    out2 = storage_z_5(out2, 0, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time, 0, 0)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Step 2: apply rotations 3-4
    out2 = apply_rot(out2, [one, one, one, z, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 3 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [one, one, one, one, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dz2 + dm2) * dx2 / dm2 * pm2,
                     0)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Step 3: apply rotations 5-6
    # Last operation: apply additional storage errors due to multi-patch measurements
    out2 = apply_rot(out2, [z, z, z, one, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 2 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [one, z, z, z, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = storage_z_5(out2, 0.5 * (dx2 + 2 * dz2 + dm2) * dm2 / dx2 * px2, 0, 0, 0, 0)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0.5 * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0.5 * px2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Step 4: apply rotations 7-8
    out2 = apply_rot(out2, [z, one, z, z, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 3 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [z, z, one, z, one], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = storage_z_5(out2, 0.5 * ((dx2 + 3 * dz2 + dm2) + (dx2 + 4 * dz2 + dm2)) * dm2 / dx2 * px2, 0, 0, 0, 0)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0.5 * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0.5 * px2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Step 5: apply rotations 9-10
    out2 = apply_rot(out2, [z, z, one, one, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [z, one, one, z, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = storage_z_5(out2, 0.5 * ((dx2 + 4 * dz2 + dm2) + (dx2 + 4 * dz2 + dm2)) * dm2 / dx2 * px2, 0, 0, 0, 0)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0.5 * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0.5 * px2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Step 6: apply rotations 11-12
    out2 = apply_rot(out2, [z, one, z, one, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [z, z, z, z, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = storage_z_5(out2, 0.5 * ((dx2 + 4 * dz2 + dm2) + (dx2 + 4 * dz2 + dm2)) * dm2 / dx2 * px2, 0, 0, 0, 0)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0.5 * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0.5 * px2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time)

    # Step 7: apply rotations 13-14
    out2 = apply_rot(out2, [one, z, one, z, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (dx2 + 4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = apply_rot(out2, [one, one, z, z, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (3 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)
    out2 = storage_z_5(out2, 0.5 * (dx2 + 4 * dz2 + dm2) * dm2 / dx2 * px2, 0, 0, 0, 0)

    # Apply storage errors for l1time code cycles
    # Qubit 1 is consumed as an output state: additional storage errors for dx2 code cycles
    out2 = storage_x_5(out2, 0.5 * px2 * (dm2 + 2 * dx2), 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time,
                       0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0.5 * px2 * (dm2 + 2 * dx2), 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time,
                       0.5 * (dx2 / dz2) * pz2 * l1time)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Step 8: apply rotation 15
    out2 = apply_rot(out2, [one, z, z, one, z], pl1 + 0.5 * lmove * pm2,
                     0.5 * lmove * pm2 + 0.5 * (4 * dz2 + dm2) * dx2 / dm2 * pm2,
                     0)

    # Apply storage errors for l1time code cycles
    out2 = storage_x_5(out2, 0, 0.5 * (dz2 / dx2) * px2 * l1time, 0.5 * (dz2 / dx2) * px2 * l1time, 0,
                       0.5 * (dz2 / dx2) * px2 * l1time)
    out2 = storage_z_5(out2, 0, 0.5 * (dx2 / dz2) * pz2 * l1time, 0.5 * (dx2 / dz2) * pz2 * l1time, 0,
                       0.5 * (dx2 / dz2) * pz2 * l1time)

    out2 = storage_x_5(out2, 0.5 * px2, 0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2,
                       0.5 * (dz2 / dx2) * px2, 0.5 * (dz2 / dx2) * px2)
    out2 = storage_z_5(out2, 0.5 * px2, 0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2,
                       0.5 * (dx2 / dz2) * pz2, 0.5 * (dx2 / dz2) * pz2)

    # Compute level-2 failure probability as the probability to measure qubits 2-5 in the |+> state
    pfail2 = np.real(1 - np.trace(np.dot(kronecker_product([one, projx, projx, projx, projx]), out2)))

    # Compute the density matrix of the post-selected output state, i.e., after projecting qubits 2-5 into |+>
    outpostsel2 = 1 / (1 - pfail2) * np.dot(np.dot(kronecker_product([one, projx, projx, projx, projx]), out2),
                                            kronecker_product([one, projx, projx, projx, projx]).conj().transpose())

    # Compute level-2 output error from the infidelity between the post-selected state and the ideal output state
    pout = np.real(1 - np.trace(np.dot(outpostsel2, ideal15to1)))

    # Full-distance computation: determine full distance required for a 100-qubit / 10000-qubit computation
    def logerr1(d):
        return 231 / float(pout) * d * float(plog(pphys, d[0]).real) - 0.01

    def logerr2(d):
        return 20284 / float(pout) * d * float(plog(pphys, d[0]).real) - 0.01

    reqdist1 = int(2 * round(optimize.root(logerr1, 3, method='hybr').x[0] / 2) + 1)
    reqdist2 = int(2 * round(optimize.root(logerr2, 3, method='hybr').x[0] / 2) + 1)

    # Print output error, failure probability, space cost, time cost and space-time cost
    nqubits = 2 * ((dx2 + 4 * dz2) * 3 * dx2 + nl1 * (
                (dx + 4 * dz) * (3 * dx + dm2 / 2) + 2 * dm) + 20 * dm2 * dm2 + 2 * dx2 * dm2)
    ncycles = 7.5 * l1time / (1 - float(pfail2))
    print('(15-to-1)x(15-to-1) with pphys=', pphys, ', dx=', dx, ', dz=', dz, ', dm=', dm, ', dx2=', dx2, ', dz2=', dz2,
          ', dm2=', dm2, ', nl1=', nl1, sep='')
    print('Output error: ', '%.4g' % pout, sep='')
    print('Failure probability: ', '%.3g' % pfail2, sep='')
    print('Qubits: ', '%.0f' % nqubits, sep='')
    print('Code cycles: ', '%.2f' % ncycles, sep='')
    print('Space-time cost: ', '%.0f' % (nqubits * ncycles), ' qubitcycles', sep='')
    print('For a 100-qubit computation: ', ('%.3f' % (nqubits * ncycles / 2 / reqdist1 ** 3)), 'd^3 (d=', reqdist1, ')',
          sep='')
    print('For a 5000-qubit computation: ', ('%.3f' % (nqubits * ncycles / 2 / reqdist2 ** 3)), 'd^3 (d=', reqdist2,
          ')', sep='')
    print('')


def add_storage_errors(state, pphys, dx, dz, dm):
    def add_errors_x(state, p1, p2345):
        return storage_x_5(state, p1, p2345, p2345, p2345, p2345)

    def add_errors_z(state, p1, p2345):
        return storage_z_5(state, p1, p2345, p2345, p2345, p2345)

    px = plog(pphys, dx)
    pz = plog(pphys, dz)

    # Add storage errors for one cycle (after applying measurement on 
    # the ancilla region). This assumes that each logical data qubit is involved
    # in at least one of the rotations for each step.
    state = add_errors_x(state, 0.5 * px, 0.5 * (dz / dx) * px)
    # Add storage errors for `dm` cycles (during applying measurement on
    # the ancilla region).
    state = add_errors_z(state, 0.5 * px * dm, 0.5 * (dx / dz) * pz * dm)
    return state

class Poistion(Enum):
    LEFT = 0,
    RIGHT = 1,


# Calculates the output error and cost of the (15-to-1) protocol in conjunction
# with the zero-level distillation with a physical error rate pphys,
# the output error rate and the failure rate of the zero-level distillation,
# distances dx2, dz2 and dm2.
def cost_of_zero_plus_15to1(pphys, pout0, pfail0, dx2, dz2, dm2, dh):
    dx = dx2
    dz = dz2
    dm = dm2
    # Introduce shorthand notation for logical error rate with distances dx2/dz2/dm2
    px = plog(pphys, dx2)
    pz = plog(pphys, dz2)
    pm = plog(pphys, dm2)
    ph = plog(pphys, dh)

    pfail = pfail0
    pl1 = pout0
    # Compute l1time, the speed at which level-2 rotations can be performed (t_{L1} in the paper)
    # l1time = max(6 * dm / (nl1 / 2) / (1 - pfail), dm2)
    l1time = 0
    for i in range(10):
        l1time += (pfail0 ** i) * (1 - pfail0) * max(dm2, i * 5)
    l1time = max(l1time, 5 / (1 - pfail0))

    def apply_rot_zero_plus_one(state, axis1, axis2, distillation_position):
        id = [one, one, one, one, one]
        assert len(axis1) == 5
        assert len(axis2) == 5
        assert axis1[0] is z
        assert axis2[0] is one
        assert distillation_position in [Poistion.LEFT, Poistion.RIGHT]

        p1 = pout0
        p2 = 0.5 * (dx2 + dz2 * 4 + dm2) * dx2 / dm2 * pm
        p3 = 0
        state = apply_rot(state, axis1, p1, p2, p3)

        p2 = 0.5 * (4 * dz2 +  dm2) * dx2 / dm2 * pm
        state = apply_rot(state, axis2, p1, p2, p3)

        # Idle Z errors during rotations:
        for (i, a) in enumerate(axis1):
            # rotations along `axis1` is position-agnostic.
            if a is one:
                continue
            succ = [j for (j, a) in enumerate(axis1) if a is z and j > i]
            if succ == []:
                continue
            next = succ[0]
            p = (ph / dh) * ((next - i) * dz + 2) * dm2
            state = apply_pauli(state, [
                a if j > i else one for (j, a) in enumerate(axis1)
            ], p)
        if distillation_position == Poistion.LEFT:
            # This means the active distillation factory is beneath qubits 2-3.
            if axis2[3] is one and axis2[4] is one:
                # There is nothing to do here.
                pass
            elif axis2[3] is one and axis2[4] is z:
                p = (ph / dh) * (dz + 2) * dm
                state = apply_pauli(state, [one, one, one, one, z], p)
            elif axis2[3] is z and axis2[4] is one:
                p = (ph / dh) * 2 * dm
                state = apply_pauli(state, [one, one, one, z, one], p)
            elif axis2[3] is z and axis2[4] is z:
                p = (ph / dh) * 2 * dm
                state = apply_pauli(state, [one, one, one, one, z], p)
                state = apply_pauli(state, [one, one, one, z, z], p)
        else:
            # This means the active distillation factory is beneath qubits 4-5.
            if axis2[1] is one and axis2[2] is one:
                # There is nothing to do here.
                pass
            elif axis2[1] is one and axis2[2] is z:
                p = (ph / dh) * 2 * dm
                state = apply_pauli(state, [one, one, z, one, one], p)
            elif axis2[1] is z and axis2[2] is one:
                p = (ph / dh) * (dz + 2) * dm
                state = apply_pauli(state, [one, z, one, one, one], p)
            elif axis2[1] is z and axis2[2] is z:
                p = (ph / dh) * 2 * dm
                state = apply_pauli(state, [one, z, one, one, one], p)
                state = apply_pauli(state, [one, z, z, one, one], p)

        # Idle Z errors:
        state = storage_z_5(state,
                            0.5 * px * dm,
                            0.5 * (dx / dz) * pz * l1time,
                            0.5 * (dx / dz) * pz * l1time,
                            0.5 * (dx / dz) * pz * l1time,
                            0.5 * (dx / dz) * pz * l1time)

        # Idle X errors during rotations:
        # Qubits involved in rotations do not suffer from idle X errors.
        # Note that qubit 1 is always involved in the axis1 rotation.
        ps = [0, 0, 0, 0, 0]
        for i in range(5):
            if axis1[i] is z or axis2[i] is z:
                ps[i] = 0
            else:
                ps[i] = 0.5 * (dx / dz) * px * dm2
        state = storage_x_5(state, *ps)

        # Idle X errors for (l1time - dm2 + 1) cycle.
        state = storage_x_5(state,
                            0.5 * px * (l1time - dm2 + 1),
                            0.5 * (dx / dz) * px * (l1time - dm2 + 1),
                            0.5 * (dx / dz) * px * (l1time - dm2 + 1),
                            0.5 * (dx / dz) * px * (l1time - dm2 + 1),
                            0.5 * (dx / dz) * px * (l1time - dm2 + 1))


        return state

    out = init5qubit

    LEFT = Poistion.LEFT
    RIGHT = Poistion.RIGHT
    # Round 1:
    # Apply rotations rotations 8[134] and 3[4].
    # Also substitute rotation 4[5] with a teleportation.
    # and hence we call apply_rot manually.
    out = apply_rot(out, [one, one, one, one, z], pl1, 0, 0)
    out = apply_rot_zero_plus_one(out,
                                  [z, one, z, z, one],
                                  [one, one, one, z, one],
                                  LEFT)
    # Apply storage errors to qubit 5.
    out = storage_z_5(out, 0, 0, 0, 0, 2 * 0.5 * (dx / dz) * pz * l1time)
    out = storage_x_5(out, 0, 0, 0, 0, 2 * 0.5 * (dz / dx) * px * l1time)

    # Round 2:
    # Apply rotations 9[145] and 13[345].
    out = apply_rot_zero_plus_one(out,
                                  [z, one, one, z, z],
                                  [one, one, z, z, z],
                                  RIGHT)

    # Round 3:
    # Apply rotations 12[12345] and 14[245].
    out = apply_rot_zero_plus_one(out,
                                  [z, z, z, z, z],
                                  [one, z, one, z, z],
                                  LEFT)

    # Round 4:
    # Apply rotations 11[135] and 15[235].
    out = apply_rot_zero_plus_one(out,
                                  [z, one, z, one, z],
                                  [one, z, z, one, z],
                                  RIGHT)

    # Round 5:
    # Apply rotations 10[125] and 5[234].
    out = apply_rot_zero_plus_one(out,
                                  [z, z, one, one, z],
                                  [one, z, z, z, one],
                                  LEFT)

    # Round 6:
    # Apply rotations 7[124] and 1[2]
    out = apply_rot_zero_plus_one(out,
                                  [z, z, one, z, one],
                                  [one, z, one, one, one],
                                  RIGHT)

    # Round 7:
    # Apply rotations 6[123] and 2[3]
    out = apply_rot_zero_plus_one(out,
                                  [z, z, z, one, one],
                                  [one, one, z, one, one],
                                  LEFT)

    # Compute level-2 failure probability as the probability to measure qubits 2-5 in the |+> state
    pfail2 = np.real(1 - np.trace(np.dot(kronecker_product([one, projx, projx, projx, projx]), out)))

    # Compute the density matrix of the post-selected output state, i.e., after projecting qubits 2-5 into |+>
    outpostsel2 = 1 / (1 - pfail2) * np.dot(np.dot(kronecker_product([one, projx, projx, projx, projx]), out),
                                            kronecker_product([one, projx, projx, projx, projx]).conj().transpose())

    # Compute level-2 output error from the infidelity between the post-selected state and the ideal output state
    pout = np.real(1 - np.trace(np.dot(outpostsel2, ideal15to1)))

    # Full-distance computation: determine full distance required for a 100-qubit / 10000-qubit computation
    def logerr1(d):
        return 231 / float(pout) * d * float(plog(pphys, d[0]).real) - 0.01

    def logerr2(d):
        return 20284 / float(pout) * d * float(plog(pphys, d[0]).real) - 0.01

    reqdist1 = int(2 * round(optimize.root(logerr1, 3, method='hybr').x[0] / 2) + 1)
    reqdist2 = int(2 * round(optimize.root(logerr2, 3, method='hybr').x[0] / 2) + 1)

    # Print output error, failure probability, space cost, time cost and space-time cost
    nqubits = 2 * ((dx2 + 4 * dz2) * dx2 + (dx2 + 4 * dz2 + dz2) * dh + (4 * dz) * dh + 16 * dm2 * dm2)
    ncycles = 7 * l1time / (1 - float(pfail2))
    print('zero-plus-1 with pphys={}, pout0 = {}, pfail0 = {}, dx = {}, dz = {}, dm = {}, dh = {}'.format(
      pphys, pout0, pfail0, dx2, dz2, dm2, dh))
    print('Output error: ', '%.4g' % pout, sep='')
    print('Failure probability: ', '%.3g' % pfail2, sep='')
    print('Qubits: ', '%.0f' % nqubits, sep='')
    print('Code cycles: ', '%.2f' % ncycles, sep='')
    print('Space-time cost: ', '%.0f' % (nqubits * ncycles), ' qubitcycles', sep='')
    print('For a 100-qubit computation: ', ('%.3f' % (nqubits * ncycles / 2 / reqdist1 ** 3)), 'd^3 (d=', reqdist1, ')',
          sep='')
    print('For a 5000-qubit computation: ', ('%.3f' % (nqubits * ncycles / 2 / reqdist2 ** 3)), 'd^3 (d=', reqdist2,
          ')', sep='')
    print('')
