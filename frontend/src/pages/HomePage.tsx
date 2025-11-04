import ButtonLink from '../components/ButtonLink';
import ImageUploadForm from '../components/ImageUploadForm';
import PageSection from '../components/PageSection';

const featureCards = [
  {
    title: 'Computer Vision Stack',
    description:
      'Build complete CV pipelines: data curation, experimentation, deployment and ongoing improvements for K3Project.'
  },
  {
    title: 'Production Infrastructure',
    description:
      'Design resilient MLOps platforms, automate training loops and ship battle-tested inference under the Nik52 brand.'
  },
  {
    title: 'Research & Prototyping',
    description:
      'Explore state-of-the-art architectures, compress models for edge and keep latency predictable for business KPIs.'
  }
];

const HomePage = () => {
  return (
    <div className="page-stack">
      <section className="hero">
        <span className="hero__badge">Nik52 | K3Project</span>
        <h1 className="hero__title">
          Minimalist AI Lab by <span className="hero__accent">Nik52</span>
        </h1>
        <p className="hero__description">
          Crafting production-grade computer vision that scales with clarity. K3Project is my lab for research,
          prototypes and real-world launches across vision, automation and analytics.
        </p>
        <div className="hero__actions">
          <ButtonLink to="/projects">View Projects</ButtonLink>
          <ButtonLink to="/contact" variant="ghost">
            Contact Nik52
          </ButtonLink>
        </div>
      </section>

      <ImageUploadForm />

      <PageSection
        subtitle="Focus"
        title="End-to-end CV leadership"
        description="I own the full lifecycle: collect data, train advanced models, deliver rock-solid inference APIs. Everything ships under the Nik52 signature."
      >
        <div className="card-grid">
          {featureCards.map((feature) => (
            <article className="card" key={feature.title}>
              <h3 className="card__title">{feature.title}</h3>
              <p className="card__description">{feature.description}</p>
            </article>
          ))}
        </div>
      </PageSection>

      <PageSection
        subtitle="Principles"
        title="The Nik52 approach"
        description="Systems thinking, measurable progress and lean architecture guide every release."
      >
        <div className="timeline">
          <div className="timeline__item">
            <h3 className="timeline__title">Measured impact</h3>
            <p className="timeline__meta">Metrics / QA / Post-processing</p>
            <p className="card__description">
              Every release is benchmarked and peer reviewed. K3Project models must pass blind evaluation before
              promotion.
            </p>
          </div>
          <div className="timeline__item">
            <h3 className="timeline__title">Lean delivery</h3>
            <p className="timeline__meta">MVP / Feedback / Iteration</p>
            <p className="card__description">
              Ship minimal versions quickly, gather signal from stakeholders and invest only in features that matter.
            </p>
          </div>
          <div className="timeline__item">
            <h3 className="timeline__title">Reliable production</h3>
            <p className="timeline__meta">MLOps / Observability / DX</p>
            <p className="card__description">
              Treat production as a system: monitoring, autoscaling, data quality guards and transparent workflows for
              every team member.
            </p>
          </div>
        </div>
      </PageSection>
    </div>
  );
};

export default HomePage;
