from grounding.study_data import balanced_sample, make_splits


def synthetic_rows():
    rows=[]
    for application in ['editor','cad','browser']:
        for screenshot in range(12):
            for target in range(3):
                rows.append({'id':f'{application}-{screenshot}-{target}',
                             'application':application,'img_filename':f'{application}/{screenshot}.png',
                             'ui_type':'text' if target%2 else 'icon'})
    return rows


def test_study_splits_are_reproducible_and_screenshot_disjoint():
    rows=synthetic_rows()
    first=make_splits(rows,dev_count=18,validation_count=36)
    second=make_splits(list(reversed(rows)),dev_count=18,validation_count=36)
    assert first==second
    assert len(first['dev'])==18 and len(first['validation'])==36
    assert not ({r['img_filename'] for r in first['dev']} & {r['img_filename'] for r in first['validation']})
    assert {r['application'] for r in first['dev']}=={'editor','cad','browser'}


def test_balanced_sampling_includes_all_available_strata_before_reusing_them():
    sample=balanced_sample(synthetic_rows(),6)
    assert len({(r['application'],r['ui_type']) for r in sample})==6


def test_previous_debug_images_cannot_enter_validation():
    rows=synthetic_rows()
    rows[0]['id']='vscode_macos_0'
    prior_image=rows[0]['img_filename']
    result=make_splits(rows,dev_count=18,validation_count=36)
    assert prior_image not in {r['img_filename'] for r in result['validation']}


def test_paired_analysis_matches_ids_and_accounts_for_all_failures():
    from grounding.study_analysis import paired_comparison

    base=[{'id':str(i),'image':f'{i}.png','correct':False} for i in range(4)]
    candidate=[{'id':str(i),'image':f'{i}.png','correct':True} for i in reversed(range(4))]
    result=paired_comparison(base,candidate,repeats=100)
    assert result['n']==4 and result['delta']==1
    assert result['cluster_bootstrap_95ci']==[1,1]
    assert result['mcnemar_exact_two_sided_p']==0.125
