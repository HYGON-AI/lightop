import torch

def test_dsa_cp_prefill_metadata():
    from lightop.kvcache import dsa_cp_prefill_metadata_plan, fused_dsa_cp_prefill_metadata
    lengths = [17, 13, 65, 31, 7, 97]
    query = [1, 2, 0, 7, 3, 65]
    table = torch.arange(8 * 128, dtype=torch.int32, device='cuda').reshape(8, 128)
    table[:, 96:] = -1
    routes = torch.tensor([6, 2, 4, 1, 7, 3], dtype=torch.int64, device='cuda')
    for cp in [2, 8]:
        for rank in range(cp):
            for page in [1, 64]:
                plan = dsa_cp_prefill_metadata_plan(lengths, query, cp, rank, 64, 43, page, True)
                out = fused_dsa_cp_prefill_metadata(torch.tensor(lengths, dtype=torch.int64, device='cuda'), table, routes, torch.tensor(query, dtype=torch.int32, device='cuda'), plan)
                enumerated = [(i, j, lengths[i] - query[i] + j + 1) for i in range(len(lengths)) for j in range(query[i])]
                local = [v for n, v in enumerate(enumerated) if n % cp == rank]
                selected = sorted(set(v[0] for v in local))
                counts = [sum(v[0] == i for v in local) for i in selected]
                cu_k = [0]
                for i in selected: cu_k.append(cu_k[-1] + lengths[i])
                cu_q = [0]
                for n in counts: cu_q.append(cu_q[-1] + n)
                kmap = {i: cu_k[n] for n, i in enumerate(selected)}
                local_map = {i: n for n, i in enumerate(selected)}
                expanded = [v[2] for v in local]
                clipped = [min(64, v) for v in expanded] + [0] * (43-len(local))
                dsa_cu = [0]
                for v in clipped: dsa_cu.append(dsa_cu[-1] + v)
                expected = dict(indexer_seq_lens=[lengths[i] for i in selected],cache_seqlens_int32=[lengths[i] for i in selected],cu_seqlens_k=cu_k,cu_seqlens_q=cu_q,dsa_seqlens_expanded=expanded,dsa_cache_seqlens_int32=clipped,dsa_cu_seqlens_k=dsa_cu,dsa_cu_seqlens_q=list(range(44)),token_to_batch_idx=[local_map[v[0]] for v in local],topk_indices_offset=[kmap[v[0]] for v in local]+[0]*(43-len(local)))
                for key, values in expected.items(): assert out[key].cpu().tolist() == values, key
                assert out['indexer_k_start_end'][0].cpu().tolist() == [kmap[v[0]] for v in local]
                assert out['indexer_k_start_end'][1].cpu().tolist() == [kmap[v[0]]+v[2] for v in local]
                page_ref = table[routes[selected], :max(lengths[i] for i in selected)]
                assert torch.equal(out['page_table_1'], page_ref)
                assert torch.equal(out['real_page_table'], page_ref[:, ::page] // page)
                flat_ref = torch.cat([table[routes[i], :lengths[i]] for i in selected])
                assert torch.equal(out['page_table_1_flattened'], flat_ref)
